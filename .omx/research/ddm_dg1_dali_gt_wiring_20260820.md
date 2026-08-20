# ddm_dg1 — the GT-lineage gate was switched off, and nobody could see it

**Task #1142** (the `na10` reopened queue + the unwired-cure P0).
**Axis:** `[macOS-CPU advisory]` lineage classification — **never a score**.
**Commit:** `17e5da4a74`. **Payloads:** `/Volumes/APDataStore/pact/ddm_dg1/retained/`.
**Effective frontier: UNMOVED.** This unit landed apparatus and one measured
artifact-identity fact. It did **not** produce a byte-closed score row, and no
claim here is a score claim.

---

## ANSWER FIRST

**The gate built to stop the GT-lineage bug class from recurring was returning a
clean `0` while 11 real findings stood.** `preflight._check_351_gt_lineage_objective_custody`
prefilters with ripgrep, which returns **absolute** paths. Handed a **relative**
`repo_root` — `Path(".")`, the form CLAUDE.md itself documents for operator audit
invocations — every `path.relative_to(repo_root)` raised `ValueError` into a bare
`continue`. 139 candidates entered the loop; **zero** reached the line scan.

Measured both ways, 2026-08-20:

| `repo_root` form | findings |
|---|---:|
| relative (`Path(".")`) | **0** |
| absolute | **11** |
| production default (absolute) | 11 |

This is `VACUITY==PASS`: a skip counted as a pass. It is the shape my own first
population measurement nearly published — I ran the detector with `Path(".")`,
got `0`, and would have reported "task 1 already complete, population clean" had
I not checked the denominator. The denominator is why this memo exists.

Fixed root-form-invariant; all three forms now agree at 11.

**Second finding:** the 2026-08-19 cure repointed `qs1.GT_POSE` at
`gt_first6_dali_n600.npy` — a file with **no registry entry**, so
`assert_gt_lineage` **refused the very table the cure installed**. The guard could
not certify the fix. Now measured and registered.

---

## §1 THE CHARTER'S PREMISE WAS STALE — validated at spawn, per apparatus law

My charter said the cure "EXISTS but is UNWIRED" with "~8 consumers still
consuming PyAV GT via `qs1.GT_POSE`". **Refuted at source.** MAIN landed
`809199d24f` on 2026-08-19: `qs1.GT_POSE` and both `mt1` defaults already point
at the DALI table, the PyAV table survives as `GT_POSE_PYAV_ADVISORY`, and the 9
`qs1` importers inherit it. Task 1 as written was already done.

Tenth-plus instance of
`[[charter_recall_validation_is_apparatus_not_volition_20260816]]`. Had I executed
the charter literally I would have re-migrated already-migrated consumers and
called it a landing.

---

## §2 THE POPULATION (M1 class-population line), with its denominator

Two instruments, deliberately, because they answer different questions.

**(a) The repo's own gate** — pose artifacts only (`gt_first6*.npy`,
`gt_cache_*.pt`); seg `gt_argmax*` is deliberately out of scope per `ddm_sp2`
(including it takes 18 files/37 sites → 93 files/153 sites, and that cache is one
established DALI lineage).

| stage | count |
|---|---:|
| ripgrep prefilter candidates | 139 |
| reaching the line scan (pre-fix, relative root) | **0** |
| reaching the line scan (post-fix) | 139 |
| **findings** | **11** |

**(b) A full census** (sister Explore arm, joined against the 58-entry
sha-keyed registry) — every live `.py` loading a pose **or seg** GT artifact:

| class | files |
|---|---:|
| **total live** GT-loading files | **407** |
| PyAV only | 283 |
| DALI only | 24 |
| MIXED (reads both, deliberately) | 10 |
| AMBIG (basename collision, unresolvable at the literal) | 43 |
| UNKNOWN / unregistered | 19 |
| archived round receipts (excluded) | 135 |

The gap between 11 and 407 is **scope, not disagreement**: the gate's regex covers
`gt_first6*` and `gt_cache_*` only. It does **not** cover `gt_argmax_n600.npy`,
`gt_posenet_pose6.npy`, `gt_n600.npz`, `gt_poses.pt`, or `gt_segnet_argmax.u8` —
which is precisely where the 43 AMBIG + 19 UNKNOWN sit. **Widening that scope is
now unblocked**, because `ddm_sp2`'s own stated widening condition is "the pose
scope reaches live-count 0", and the 11 remaining findings are all label-only
(§3). Recorded as owed, not done.

**Correction to a premise in my own charter:** *basename is not lineage.*
`gt_first6_n600.npy` exists at two shas with **opposite** lineages — `1f2fe6d1…`
(po1/qs1 roots) is **DALI**, `82ed61ce…` (the pz4 root) is **PyAV**. The rule
"`gt_first6_n600.npy` = PyAV" is true only under one root. `gt_argmax_n600.npy`
has three shas across two lineages. This is exactly why the registry is
content-addressed, and why any name-keyed migration would have been wrong.

---

## §3 THE 11 FINDINGS ARE LABEL DEFECTS, NOT OBJECTIVE DEFECTS — measured

Resolved every flagged artifact against the registry by sha:

| artifact | sites | registry lineage | verdict |
|---|---:|---|---|
| `gt_cache_600_official_ada.pt` | 10 | **DALI_NVDEC** | correct objective, **undeclared** |
| `gt_cache_av.pt` (`ddm_jg1_seg_solve.py:89`) | 1 | PYAV_YUV420_TO_RGB | deliberate two-lineage differencing instrument |

**No wrong-objective solve is hiding in the 11.** Every one reads the DALI
authority already, or reads PyAV on purpose to measure the fork. The fix is a
declaration (`# GT_LINEAGE_OK:<rationale>` or a route through the registry), not a
repoint. I did **not** apply those 11 declarations: they are cosmetic relative to
the two structural defects, and adding 11 waiver comments would drive the gauge to
0 by *annotating* rather than by *curing* — the instrumentation-instead-of-reality
trap `tac.gt_lineage`'s own docstring rejects. Left as an explicit owed row.

---

## §4 THE CURE'S TARGET IS NOW VERIFIABLE — measured, $0

`gt_first6_dali_n600.npy` postdates the 2026-08-17 registry build, so it carried
**no sha row** and `assert_gt_lineage` raised `GtLineageUnknown` on it.

Measured against the #906 producer-declared rulers (n600, fp64 reduction):

| comparison | MSE |
|---|---:|
| **cure target vs DALI ruler** | **0.0 — `array_equal` TRUE, bit-identical** |
| cure target vs AV ruler | 1.40613249e-04 |
| PyAV table vs AV ruler | 4.8898e-12 |
| PyAV table vs DALI ruler | 1.40615094e-04 |
| **C = MSE(dali table, pyav table)** | **1.40615094e-04** |
| `na10` published C | 1.406151e-04 |

Two things fall out. The cure's target **is** the DALI ruler's pose table, byte
for byte — evidence grade `EMPIRICAL_EXACT_MATCH`, the strongest the registry
has. And **C reproduces `na10`'s constant to 7 digits by an independent route**
(a fifth route, after pi2's decomposition, na10's 3-body offset, the direct table
MSE, and the registry's `EMPIRICAL_NEAREST_RULER_POSE_MSE`).

Registered through the **canonical producer**, not by hand: `ddm_gl1_gt_lineage_census.py`
gained `--classify-path` + `--merge-into`, an additive merge that refuses if a
shared sha reclassifies, unions `known_paths`, preserves the original census
denominators, and records an `amendments` entry so a reader can always tell a full
census from a census plus patches. Registry 58 → 59; the merge is idempotent
(re-run: 59 → 59). A hand-edited registry would have been a parallel authority —
the exact failure the module exists to prevent.

Verified after: the DALI table **certifies** on the CUDA axis; the PyAV table is
**refused** on it.

---

## §5 THE DELTA GUARD — the layer the jg4 incident proved missing

Two guards already existed and **neither could see the jg4 failure**:
`assert_gt_lineage` keys on a GT **file**, and the T4 leg was a number read out of
a contest report with no local file at all; `assert_single_lineage` checks the
span **within one instrument**, not **across two separately-measured legs**.

jg4 subtracted a PyAV-lineage advisory seg (0.0003244) from a DALI-lineage T4 base
(0.00030309), read +2.1e-05, and called a working candidate net-negative. The
base's own advisory reading is 0.00043336 = **1.430×** its T4 value: the "effect"
was the lineage fork. Same-instrument the candidate had **improved** (−1.090e-4).
A candidate projecting S ≈ 0.1467 was nearly killed by one subtraction.

`tac.local_contest_instruments` now carries `assert_comparable_legs` +
`receipt_delta`, checking the full comparability tuple **(axis, gt_lineage, pairs,
sampling)** — not lineage alone, because a 600-pair leg minus a 96-pair leg is
also a fork, and prefix bias runs 2.54–4.21× *harder* on pose and 0.95–0.97×
*easier* on seg, so it inverts sign per axis.

Design decisions worth naming:

* The refusal **quotes the mechanism** (seg multiplicative ~1.4425×, pose additive
  +1.4061e-04, per-pair ratio span 0.887–1,627). A refusal that does not teach
  gets waived away.
* The pose message says **ADDITIVELY** and refuses to offer a multiplier —
  `na10` measured that the multiplicative form is the refuted one.
* The waiver takes a **substantive rationale** (placeholders rejected) and is
  scoped to the two same-instrument checks; it deliberately does **not** waive
  the population check. One flag that switches off every refusal is the
  over-broad-waiver shape.

**Positive control, executed in the test suite:** the same-lineage delta
reproduces jg4's published −1.0896e-4 to 2e-7. A guard that simply refused
everything would still fail that test.

22 tests; 212 pass across the lineage suite. The vacuity test asserts the
**denominator**, not the finding count — asserting the count would start failing
the day the repo is genuinely cured, which is success.

---

## §6 THE RE-MEASURES — NOT RUN. Honest state and sealed fire-orders.

Charter task 3 asked for qs3 / sq2 R8 / ps135b / ps1u re-measures. **I ran none of
them.** Two structural defects surfaced en route — a dead gate and an
uncertifiable cure target — and both are *upstream of every one of those
re-measures*: each would have produced numbers a reader would compare against a
base, through the comparison path that was unguarded, using a table the registry
could not certify. Fixing the instrument before spending it on rows is the order
the measurement-first discipline actually implies. Saying so plainly rather than
reporting partial rows as progress.

State of each, with what is now unblocked:

| row | blocker at spawn | state now | cost |
|---|---|---|---:|
| `qs3` attribution (na10 #6) | "matched T4 GT argmax field absent" | `qs3` memo's own header says **QS4 superseded this** — the hash-pinned matched GT field landed 2026-08-13. Blocker likely already dissolved; needs verification against `ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy` (sha `91d3ff11`, registry-confirmed DALI) before any attribution run | $0 |
| `sq2` R8 | carrier re-solve | unblocked; long | $0, ~4 h |
| `ps135b` / `ps1u` (na10 #2/#3) | PyAV objective **and** PyAV pair-selection (`top_mass_pairs`: Spearman 0.122 vs DALI, top-30 overlap 1/30) | objective now DALI via `qs1`; **the selector is still PyAV-ranked** and is a separate repoint nobody has made | $0 solve + **$0.16** T4 |
| `na2` ×4 + `na5` ×5 | source parity / absent harness | blocker class dissolved (jg1/up2 $0 instruments) | $0, gated on rebuilding harnesses |

**Fire-order for MAIN — the one I would run first:** `ps1u`'s `top_mass_pairs`
selector. It is the last live PyAV-lineage limb of the cured objective: the solve
now optimises the DALI objective but still *chooses which pairs to spend on* by
PyAV residual, and those two rankings agree at Spearman 0.122 with 1/30 top-30
overlap. That is a wrong-object defect of the same genus as the one just cured,
sitting one function away from it, and it is $0 to repoint. **No paid dispatch
fired; none is requested by this memo.**

---

## §7 WHAT THIS COST, AND THE GENUS

The vacuity defect is `[[m50]]` (VACUITY==PASS — skip counted as green, cure:
report the denominator) crossed with `[[structural_beats_procedural_and_the_detector_that_zeroes_on_the_cure_20260803]]`.
The uncertifiable cure target is `[[m56]]` (unwired-but-built) at one further
remove: the cure was wired, but the *guard for the cure* could not reach it.

The sharpest lesson is about my own first measurement. I ran the repo's own
detector, got `0`, and was one step from writing "population clean, task
complete". The only thing between that and the truth was asking **what is the
denominator** — 139 candidates in, 0 scanned. A gate reporting zero and a gate
that never ran are the same string at the call site.

Sisters: `[[bug_ladder_bugs_classes_families_metabugs_20260819]]` (SILENT-INSTRUMENT
and WRONG-OBJECT families, both live here) ·
`[[measured_object_vs_named_object_20260816]]` (the basename-is-not-lineage
correction) · `[[advisory_gate_cross_instrument_false_refusal_20260819]]` (the
incident the delta guard extincts).

---

## §8 OWED, with owners

1. **Widen the #351 scope to seg** (`gt_argmax*`, `gt_n600.npz`,
   `gt_segnet_argmax.u8`) — `ddm_sp2`'s stated widening condition is met. The 43
   AMBIG + 19 UNKNOWN files pass unchallenged today; 8 of them take their argmax
   cache from a CLI arg with no static root, so lineage is caller-determined and
   nothing in the file pins it. **Those 8 are the genuinely dangerous ones.**
2. **Declare the 11** label-only findings (waiver or route). Drives the pose scope
   to live-count 0 and makes the STRICT flip available.
3. **Flip #351 to STRICT** once (2) lands — live count 0 is the strict-flip
   atomicity condition.
4. **Repoint `ps1u`'s `top_mass_pairs`** to DALI ranking (§6 fire-order).
5. **Adopt `receipt_delta` at the comparison sites.** The guard is built and
   tested; until callers route through it, it is itself an unwired cure — the very
   pattern this memo is about. It reaches nothing on its own.
6. `tools/run_ddm_pk2_pr130_surface_fit.py` (PyAV, 2026-08-08),
   `tools/measure_mdl_ms_complex_k_lower_bound.py`,
   `tools/render_witness_morse_smale_viz.py` all predate the cure and read PyAV.
   The last one's docstring asserts it avoids PyAV while consuming a
   registry-confirmed PyAV cache — a claim/registry disagreement worth resolving,
   not silently trusting.

---

## Observability surface

Inspectable per layer (stage-by-stage detector denominator, reproducible via the
snippet in §2) · decomposable per signal (per-artifact MSE vs both rulers) ·
diff-able across runs (registry is sha-keyed; merge is idempotent) · queryable
post-hoc (`retained/ddm_dg1_gt_lineage_verification_receipt.json` + both pose
tables retained, 14,528 B each, shas `8d5cfa83df55` / `82ed61ce6a11`) · cite-able
(commit `17e5da4a74`) · counterfactual-able (`--classify-path` re-classifies any
artifact against the rulers on demand).
