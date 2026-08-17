# ddm_gl2 — the second GT population: `gt_segnet_argmax.u8` is PyAV, and the 20,673 px is the known lineage split

**Axis `[macOS-CPU advisory]` throughout. Lineage classification is never a score. This unit did
not lower the score and does not claim to.**

**Scope, stated first so nothing here reads bigger than it is: this is a COVERAGE-GAP CURE of an
EXISTING genus, not a discovery.** `ddm_pi2` found the genus (one instrument, two decode lineages).
`ddm_gl1` built the registry and the fail-closed guard. `ddm_gl2` found that both of gl1's discovery
legs carry an extension allow-list, measured the population that allow-list could not see, and
closed it. The mechanism was already named; what was missing was that a second population existed
at all.

---

## §1 THE ANSWER

`ddm_a1s` §7 measured that `gt_segnet_argmax.u8` and the qs3 `gt_argmax_n600.npy` differ on
**20,673 of 117,964,800 sites = 0.017525 S units**, and said plainly it had not established the
mechanism, naming batch-shape tie-breaking (`et4`), a different decode path, and thread count as
live candidates.

**It is the decode path. HYPOTHESIS CONFIRMED.**

| artifact | sha256 | bytes | lineage | evidence |
|---|---|---:|---|---|
| `targets_n600/gt_segnet_argmax.u8` | `36c6be718916…` | 117,964,800 | **PYAV_YUV420_TO_RGB** | `PRODUCER_DECLARED` + nearest-ruler |
| `targets_n16/gt_segnet_argmax.u8` | `7646794a39c8…` | 3,145,728 | **PYAV_YUV420_TO_RGB** | `PRODUCER_DECLARED` + **16/16 frames bit-identical to the AV ruler** |
| qs3 `gt_argmax_n600.npy` (a1s's reference) | `91d3ff11…` | 117,964,928 | DALI_NVDEC | gl1, nearest-ruler |

Measured against the two Modal-#906 rulers (PR130's `build_gt_cache_official.py` run twice in ONE
container on ONE Tesla T4, `--dataset av` then `--dataset dali`, so they differ ONLY by decoder):

```
gt_segnet_argmax.u8 (n600)  vs AV/PyAV ruler:        1 differing site
gt_segnet_argmax.u8 (n600)  vs DALI ruler:      20,672 differing sites
```

**One site versus twenty thousand six hundred seventy-two.** The `.u8` was built on macOS CPU on
2026-06-10; the rulers were built on a Tesla T4 on 2026-08-09. Across two months, two hosts, and
CPU-versus-CUDA numerics, the same decode path reproduces to **1 site in 117,964,800**.

---

## §2 WHY THE FALSIFIER FAILS — the ruler geometry settles it

The falsifier I was given was: *if the disagreement is inconsistent with a decode-lineage split —
if it looks like a tie-break or batch-shape artifact, or if the `.u8` matches neither ruler — say
so.* It does not, and here is the measurement that decides it.

Both rulers came from **one container, one T4, one clock, one code path**; the only variable was
the `--dataset` argv. So batch shape, thread count, driver, and library versions are held FIXED
ACROSS the two rulers. A `.u8` whose disagreement came from batch shape or thread count would
therefore land at roughly the SAME distance from both rulers — those axes do not separate them.
Instead it lands **20,672× closer to one than the other**, along precisely the axis the rulers
were built to isolate.

Three independent corroborations, each already in the registry or measured here:

1. **The separation constant.** gl1 measured the DALI↔PyAV separation at 20,670 / 20,671 / 20,672
   sites across three different artifact pairs. The `.u8`-vs-`91d3ff11` figure is **20,673**, i.e.
   the separation plus the two artifacts' own within-family drift (1 site and 3 sites). It is the
   same constant, not a new number.
2. **Producer receipt, BOUND to these exact bytes.** `targets_meta.json` publishes an
   `argmax_class_histogram`; recomputing those five fractions from the candidate bytes reproduces
   them with **max |Δ| = 0.0** at full float precision. Only then is the receipt evidence about
   THESE bytes rather than a JSON that happens to sit in the same directory. The bound receipt
   names `tools/lever_b_build_score_native_targets.py`, which imports `frame_utils.yuv420_to_rgb`
   and *raises* on any non-CPU device. `PRODUCER_DECLARED` is the top rung of gl1's ladder, and
   before this unit only the two #906 rulers held it.
3. **The n16 sibling is bit-identical.** 16/16 frames of `targets_n16/gt_segnet_argmax.u8` are
   bit-identical to the corresponding AV-ruler frames, 0/16 to DALI. Exact content identity, no
   margin to argue about.

**What would still overturn this:** a producer receipt for `91d3ff11…` proving it too was decoded
with PyAV, which would move the disagreement onto some other axis. gl1 classified `91d3ff11…`
empirically (3 sites from the DALI ruler), not from a receipt, so that door is open. I did not find
such a receipt and I did not look exhaustively — I searched the artifact's own directories only.

---

## §3 THE 20,673 PIXELS, CHARACTERISED

Reproduced exactly: **20,673 / 117,964,800**, agreement 117,944,127 = 99.9824752807%,
d_seg 1.752471923828125e-04, **0.017525 S units at the 100× seg weight**.

**Spatial.** All **600 of 600 frames** disagree — min 14, max 103, mean 34.5, median 33 sites per
frame. Every one of the 512 columns carries mass; only **141 of 384 rows** do, and they sit inside
the single band **rows 155–297**. Rows 0–154 and 298–383 carry **exactly zero**. Rows 283–288 alone
hold **20.6%** of the total. (That the quiet regions are sky above and the static ego-hood below is
an *inference* from the canonical class row-extents, not something this mask measures.)

**On the boundary, not in the interior.** The discriminating measurement, with its denominator
beside it:

| quantity | value |
|---|---:|
| sites adjacent to a class change, whole population | 2,551,382 / 117,964,800 = **2.163%** |
| sites adjacent to a class change, among the 20,673 | 20,401 / 20,673 = **98.684%** |
| **enrichment** | **45.63×** |

A photometric difference can only flip a pixel whose top-two logits are already close, and those
live on the codim-1 argmax boundary. 98.7% boundary occupancy at 45.6× enrichment is that
signature. Interior mass would have pointed elsewhere.

**Per-class confusion** (row = class in the PyAV `.u8`, column = class in the DALI `.npy`; canonical
comma10k order `0 Road · 1 Lane · 2 Undrivable · 3 Movable · 4 MyCar`):

|  | →0 | →1 | →2 | →3 | →4 |
|---|---:|---:|---:|---:|---:|
| **0 Road** | — | 2,912 | 2,770 | 650 | 2,771 |
| **1 Lane** | 2,785 | — | 17 | 10 | 12 |
| **2 Undrivable** | 2,898 | 5 | — | 1,083 | 0 |
| **3 Movable** | 694 | 9 | 985 | — | 0 |
| **4 MyCar** | 3,053 | 13 | 0 | 6 | — |

Road is the hub: it is one side of 9,103 sites read from the PyAV field and 9,430 read from the
DALI field — consistent with the measured seg graph (`pc2`: Road in 87.8% of flips). The two
directions are near-balanced (9,103 vs 9,430), so this is not a systematic shift of one class into
another; it is a small perturbation nudging near-tie pixels both ways across shared edges.

---

## §4 THE COVERAGE HOLE, MEASURED AT SOURCE

Not inferred from the miss — read out of gl1's own committed code and its own retained log.

1. `experiments/ddm_gl1_gt_lineage_census.py::enumerate_code_referenced` matches
   `gt_[A-Za-z0-9_]*\.(?:npy|npz|pt|pth)`. `.u8` and `.f16` are not in the alternation.
2. `_extract_fields` returns `unhandled suffix {path.suffix}` for every other container.
3. gl1's retained `logs/disk_sweep.txt` — 66 files, 23.2 GiB — has the suffix histogram
   **26 `.npy` + 9 `.npz` + 31 `.pt`** and **zero** `.u8`, **zero** `.f16`.

So the hole is not one forgotten line. **Both discovery legs carry the same extension allow-list,
and the extractor reinforces it.** An allow-list fails OPEN on every container nobody anticipated:
the population is not merely unclassified, it is invisible, and "found nothing" reads exactly like
"there is nothing here."

A raw memmap also needs different handling, not just a wider regex. A `.npy` carries its own shape
and dtype; a `.u8` carries nothing at all — the shape lives only in the reader's source. This unit
therefore infers the frame count from the byte count against the known seg geometry and REFUSES
when it does not divide, rather than assuming one.

---

## §5 WHAT I REGISTERED, AND WHAT I REFUSED TO REGISTER

Merged into `src/tac/gt_lineage_registry.json` by measurement, sha256-keyed, additive-only — an
existing gl1 row is never overwritten, only its `known_paths` widened, so a second census cannot
silently reclassify what the first measured.

**Registered with a resolved lineage (2):** the two `gt_segnet_argmax.u8` fields above.

**Registered as UNKNOWN, deliberately (8) — recorded, and refused:**

| artifact | bytes | label | why unresolved |
|---|---:|---|---|
| `targets_n600/gt_segnet_margin.f16` | 235,929,600 | `UNKNOWN_UNCOMPARABLE` | carries seg geometry but stores a float top1−top2 margin, not class labels; the rulers store argmax labels, so there is nothing to compare |
| `targets_n16/gt_segnet_margin.f16` | 6,291,456 | `UNKNOWN_UNCOMPARABLE` | same |
| `teacher_logits_n600/gt_segnet_logits.f16` | 1,179,648,000 | `UNKNOWN_UNCOMPARABLE` | 5-class logits, same reason |
| `taskspace_fresh_scorer_planes_n600_20260726/stage_{00..04}/gt_poses.f32` ×5 | 2,880 each | `UNKNOWN_AMBIGUOUS` | (120, 6) pose tables; **0/120 rows match either ruler exactly**, and for a bare pose table there is no cheap second test. **Measurement owed.** |

These five `gt_poses.f32` files came from the **disk sweep**, not from a code literal — the raw
sweep of both SSD tiers plus the repo returned 1,702 raw-suffix files, of which 10 carry a `gt_`
basename. They are recorded so they are no longer invisible; their lineage is not claimed. Naming
one would have required asserting a decode path I did not measure, which is the thing this whole
line of work exists to stop.

Recording an UNKNOWN buys **visibility, never usability**. `assert_gt_lineage` still refuses every
unresolved row, and each carries the claim boundary *"Lineage NOT established."* Separating those
two properties is deliberate: conflate them and people either register junk to silence a gate or
leave real artifacts unrecorded to avoid mislabelling them.

Registry after the merge: **58 entries** (48 from gl1 + 10 here) — 32 PyAV, 6 DALI, 11
`UNKNOWN_AMBIGUOUS`, 9 `UNKNOWN_UNCOMPARABLE`.

**Refused outright: nothing was labelled from an unmeasured premise.** No lineage in this merge
rests on a name, a directory, or a neighbouring file. Six files share the basename
`gt_argmax_n600.npy` and `ddm_pi2` verified exactly one; the two `.u8` files here share a basename
and are different bytes with their own rows. That is the whole point of content addressing, and
re-committing the laundering error inside the cure would have been the fake.

---

## §6 THE CURE (landing 1) AND THE GATE (landing 2)

**Landing 1 — the population is now visible and the name-keyed surface is closed.**

* `src/tac/gt_lineage_registry.json` — the measured rows, sha256-keyed.
* `src/tac/measurement_integrity.py` — `GT_ARGMAX_TOKENS` now carries an explicit boundary:
  **detection vocabulary only, a name is not an identity and never a lineage**, with the measured
  numbers and a pointer to the content-addressed answer in `tac.gt_lineage`. Its replacement,
  `find_gt_artifact_literals`, returns PATH literals — a path resolves to bytes, bytes to a sha256,
  a sha256 to a recorded lineage; a bare name resolves to nothing.
* **The polarity is inverted.** `NON_FIELD_SUFFIXES` is a DENY-list of things that are
  definitionally not tensor fields (`.py`, `.json`, `.md`, receipts, config). Every other suffix —
  including containers nobody has invented yet — lands in the must-be-registered population
  automatically. Unknown now fails CLOSED.

**Landing 2 — `tac.gt_lineage.assert_gt_population_registered`**, which REFUSES when any
ground-truth artifact reachable from our own instruments has an unregistered sha256. Its escape
hatch, `allow_sha256`, is keyed by digest and never by name, because a name-keyed allow-list would
reintroduce the exact defect it excuses. `unregistered_gt_artifacts` is the paired gauge, and it
does not zero out on its own cure: adding a lineage *declaration* to an already-registered artifact
moves it by exactly zero; it moves only when a GT artifact enters or leaves a reachable read path.

**Executed positive control** (`--gate-positive-control`, receipt `GL2_GATE_POSITIVE_CONTROL.json`):

| control | expected | observed |
|---|---|---|
| unregistered raw `.u8` reachable from a reader | gate RAISES, naming the digest | **PASS** |
| unregistered `.zz9` — a container nobody has defined | gate RAISES (deny-list polarity) | **PASS** |
| same file, DIGEST registered | gate silent | **PASS** |

**Live count, re-derived by running the gate — not asserted, not copied:**

| when | unregistered GT artifacts reachable from code |
|---|---:|
| before the merge | **5** |
| after the merge | **0** |

Scope of that gauge, stated so the zero is not read as bigger than it is: it counts artifacts
reachable from a **code path literal** resolved through the declared search roots. The five
`gt_poses.f32` files reached only by the disk sweep were never in it — they are registered anyway,
because visibility should not depend on which discovery leg happened to find something.

Tests: `src/tac/tests/test_gt_lineage_raw_population.py`, 29 passing, including
`test_registering_a_DIFFERENT_file_does_not_clear_the_gate` (same basename, same byte count,
different bytes → still refused) and `test_allow_list_is_keyed_by_digest_not_by_name`. gl1's own
27 tests still pass; Catalog #392 still passes STRICT.

**No new catalog number was claimed.** `tools/claim_catalog_number.py peek` returns 408 and Catalog
#299 `check_catalog_quota_under_400` is wired STRICT, so a new `preflight_all` entry is refused
until a gate is retired — gl1 recorded the same blocker as its debt item 9. Per CLAUDE.md's
two-landing rule the second landing may be *"a NEW STRICT preflight gate **OR** a canonical-helper
invariant that refuses re-introduction"*; this is the invariant. The preflight wire-in stays owed,
with the blocker reduced to one step: the callable, its controls, and a live count of 0 are ready,
so the next owner needs only the catalog claim.

---

## §7 BLAST RADIUS — 15 readers, all on the PROXY axis

`gt_segnet_argmax.u8` (n600) has **15** readers, the n16 sibling **12**, counting only files that
name it and excluding this unit's own census tool. Classified by what they can emit:

| class | files | can it produce a contest-axis score? |
|---|---|---|
| explicitly `WITNESS_DSEG_FEASIBILITY_ONLY` | `lever_b_score_native_argmax_smoke.py`, `witness_capstone_deepmath_smoke.py`, `score_native_build_byte_closed_candidate.py` | No — marker + tag required by Catalog #392 |
| advisory-tagged probes / builders | `build_segnet_teacher_logits.py`, `build_static_region_prior.py`, `lever_b_build_score_native_targets.py`, `lever_b_train_generator_checkpoint.py`, `lever_c_train_conv_pair_decoder.py`, `probe_regmax_family.py`, `score_native_first_candidate_smoke.py`, `score_native_lowres_appearance_probe.py`, `score_native_assemble_pose_carrier_candidate.py`, `probe_taskspace_witness_feasibility.py`, `measure_symbolic_topological_partition_mdl.py` | No |
| touch `archive.zip` | `score_native_build_byte_closed_candidate.py`, `score_native_assemble_pose_carrier_candidate.py` | No — both carry the feasibility marker / advisory tags |

**Every d_seg computed against this field is a PROXY** — generator argmax versus GT argmax, with no
reconstruction operator `R` and no SegNet re-segmentation — which `tac.measurement_integrity`
already classifies as feasibility-only and never a score. So the 0.017525 S units is **not** an
error in any contest-axis number.

**Where it does bite:** cross-comparison. Any number computed against the PyAV `.u8` carries a
0.017525 S offset relative to any number computed against a DALI cache. That is 59.2% of the whole
scored seg term and 1.83× the remaining −0.0095973 gap — large enough to invert a verdict between
two arms that each thought they were reading "the GT argmax". `ddm_a1s` was right to refuse the
comparison; it now has the reason.

---

## §8 VERDICT SCOPE

- **FORMULATION** — the extension-allow-list mechanism, the deny-list polarity, and the
  producer-receipt binding rule (recompute a published fingerprint before believing a sibling
  file). Properties of how discovery is written, not of any archive.
- **INSTANCE** — every sha256, the 20,673 / 20,672 / 1 site counts, the 45.63× boundary enrichment,
  the 15-reader blast radius, and the 5→0 live count. Properties of this repo and these SSD tiers
  as of 2026-08-17; re-measure when caches or call sites change.
- **`0.mkv`-specific** — the ~20,671-site DALI↔PyAV separation and the 0.017525 S figure are
  properties of one clip and two decoders, inherited from pi2 §6.
- **NOT established** — no producer receipt for `91d3ff11…` (its DALI label remains gl1's empirical
  nearest-ruler classification); no re-pointing of any of the 15 readers, which is an owner
  decision about which lineage each one needs; no exact-eval row; no score.

**Apparatus debt handed on, each with an owner and a fire-condition:**

| # | work | owner | fire-condition |
|---|---|---|---|
| 1 | Wire `assert_gt_population_registered` into `preflight_all` | apparatus owner | when a catalog retirement candidate is identified (quota is at 408) |
| 2 | Decide which lineage each of the 15 readers needs, then assert it per read | the arm that next touches a lever_b / score_native surface | on next edit to any of them |
| 3 | Fold gl1's and gl2's discovery onto ONE enumerator so a third census cannot re-fork the rule | whoever proposes a third census | when a third GT census is proposed — refuse it, merge instead |
| 4 | Obtain or refuse a producer receipt for `91d3ff11…` | whoever next relies on it as a DALI reference | before any claim that needs `PRODUCER_DECLARED` on it |

---

## §9 RETAINED PAYLOADS (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_gl2_gt_lineage_u8_coverage_20260816/`. VertigoDataTier is full
and read-only — read from, never written to.

| artifact | bytes | sha256 | what it is |
|---|---:|---|---|
| `disagreement/disagreement_mask_packbits.u8` | 14,745,600 | `6c31cd2f0ec79e83…` | the FULL (600,384,512) boolean disagreement mask, `np.packbits`, C order — the field itself, not its count |
| `disagreement/disagreement_per_frame.npy` | 4,928 | `0d4a8f386f6f52e1…` | per-frame differing-site counts, 600 rows |
| `disagreement/disagreement_per_row.npy` | 3,200 | `cea222899eb1a4d0…` | row profile, 384 rows |
| `disagreement/disagreement_per_col.npy` | 4,224 | `4a310f96c4628baa…` | column profile, 512 cols |
| `disagreement/disagreement_class_confusion.npy` | 328 | `5da309f11aff6c82…` | the 5×5 confusion above |
| `GL2_RAW_GT_LINEAGE_CENSUS.json` | — | — | every classified artifact: sha256, both ruler distances, receipt binding, readers, the disagreement block |
| `GL2_GATE_POSITIVE_CONTROL.json` | — | — | the three executed controls + the re-derived live count |
| `gt_lineage_registry.PRE_GL2.json` | — | — | the registry as gl1 left it, so the merge is auditable rather than asserted |
| `scratch/raw_sweep_all.txt` | — | — | the raw-suffix disk sweep of both SSD tiers + the repo (1,702 files; 10 with a `gt_` basename) |

Committed: `experiments/ddm_gl2_raw_gt_lineage_census.py`, `src/tac/gt_lineage.py`,
`src/tac/gt_lineage_registry.json`, `src/tac/measurement_integrity.py`,
`src/tac/tests/test_gt_lineage_raw_population.py`.

Consumed unmodified and custody-verified: the #906 rulers `gt_cache_dali.pt` (`a91d9825…`) and
`gt_cache_av.pt` (`837b5852…`) — the census refuses to classify anything if their sha256 do not
match — and qs3 `gt_argmax_n600.npy` (`91d3ff11…`).

**Own-vehicle frontier: hv1 ep0634, S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4 n600]` —
UNMOVED.** This unit repaired an instrument's memory. It did not lower the score.
