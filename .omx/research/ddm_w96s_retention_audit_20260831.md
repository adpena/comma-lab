---
arm: ddm_w96s_retention_audit
utc: 2026-08-31
axis: "[macOS-CPU filesystem measurement + source read; no scorer, Metal, training, Modal, n600, or contest evaluation]"
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
verdict_scope: "STORAGE CUSTODY — the realized w96b two-seed retention tree, and the leg-4 classification of two named AP trees. Closes no scientific family and reopens none."
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_w96s — the 22.32 GB was free space, not demand; w96b already fired; the storage ask was withdrawn six hours before I was spawned

## Answer first

Three findings, in descending order of how much they change the problem.

1. **The mandate's premise is false. `w96b` is not unfired.** It ran both seeds on
   2026-08-27 and **CLOSED the family** at the chartered fork rule. Seed 20260815 ratio
   **1.186×**, seed 20260816 ratio **1.5796×**, both inside the `<2×` closure zone against a
   `≥5×` LIVE gate. There is nothing to unblock.
2. **The 45.52-vs-22.32 inversion is confirmed at source, and my mandate had it backwards.**
   `22,319,071,232 B` is the **recorded free bytes** on APDataStore at w96a's 08-26 preflight.
   It was never a demand. `45,521,567,744 B` is the projected two-seed demand
   (`36,931,633,152`) plus an 8 GiB reserve. Both numbers are in the w96a memo and the w96b
   receipt.
3. **No operator storage decision is owed, and MAIN already established that today.**
   `ddm_rb1_pose_arithmetic_closure_and_storage_no_consumer_20260831.md` (08-31 01:43)
   adjudicated all three legs of the storage-gated fire DEAD and withdrew both asks verbatim:
   *"Therefore the AP reclaim has no live consumer. The 60.38 GB (four-config) and 14.3 GB
   (one-config) asks are both withdrawn. No operator storage decision is owed."* My arm was
   spawned after that memo landed.

What my arm adds that no prior memo holds: the **realized** decomposition of the tree that run
produced, and the retention law that falls out of it.

**The two-seed w96b tree is 29,662,904,320 B. Its verdict-critical content is 37,198,539 B.
The ratio is 797×.** My prior-law prediction said verdict-critical retention would be under
5 GB. It is under 5 GB by a further factor of 134. The prediction is confirmed, harder than I
wrote it.

The exact pointer is unmoved. No byte was deleted, moved, compressed, or written to either SSD.

## 1. The 45.52 / 22.32 reconciliation

Both numbers are real. They play different roles, and my mandate swapped them.

| quantity | value (B) | GB | role | receipt |
|---|---:|---:|---|---|
| AP free at w96a preflight | 22,319,071,232 | 22.32 | **free space** | `w96a` memo ln 19; `W96B_BUILD_AND_STORAGE_RECEIPT.json` key `recorded_available_bytes` |
| Projected two-seed demand (pre-dedup) | 36,931,633,152 | 36.93 | demand | `w96a` memo ln 18 |
| Reference one-seed demand | 18,465,816,576 | 18.47 | demand/2 | receipt key `reference_one_seed_allocated_bytes` |
| Demand + 8 GiB reserve = **fire trigger** | 45,521,567,744 | 45.52 | gate threshold | `w96a` fire order item 1 |
| Two-seed demand **post-dedup** | 24,979,443,712 | 24.98 | demand, CAS-aware | receipt key `two_seed_65epoch_post_dedup_allocated_bytes` |
| **Realized on disk (MEASURED by me)** | **29,662,904,320** | **29.66** | actual | `du -sk`, this arm |

`45,521,567,744 − 36,931,633,152 = 8,589,934,592` = exactly 8 GiB. The arithmetic closes.

Two corrections to the record beyond the inversion:

- The charter falsifier my mandate quoted as *">22.32 GB → storage routes to #1165"* compares a
  **demand** against that **free-space** figure. Post-dedup demand 24.98 GB exceeded free
  22.32 GB by 2,660,372,480 B, so the falsifier fired and the receipt recorded
  `falsifier_route: #1165`. It fired correctly. It did not mean "demand is 22.32".
- The retention measurement the charter required **did run**. It is
  `tac_content_addressed_cohort_inventory.v1`, a source-backed exact SHA-256 inventory over 35
  OFF evaluation trees, `source_tree_count: 35`, `measurement: … each source tree hashed once;
  no source payload changed`. This is **not** a #878 not-recorded case.

Receipt: `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/W96B_BUILD_AND_STORAGE_RECEIPT.json`,
33,416 B, sha256 `6fac1bd8c8c40f9fc97df57b341b0b99c010105ecbdf12c2495304e7491cd42b`.

### Why the projection overshot

Projected 36.93 GB, realized 29.66 GB — the projection ran 24.5% high. The gap is dedup the
projection could not fully anticipate. I measured the achieved factor from the 28 retention
manifests: **27,145,586,161 B logical → 18,866,575,375 B unique = 1.4388×**, saving
8,279,010,786 B. The store holds 2,301 unique objects behind 6,846 chunk references across
3,164 logical files.

## 2. Demand decomposition — what the 29.66 GB actually counts

`du -sk` × 1024, APDataStore, 2026-08-31.

| component | bytes | GB | % of tree | kind |
|---|---:|---:|---:|---|
| **TOTAL tree** | **29,662,904,320** | **29.663** | 100.00 | |
| `content_addressed_store/` | 19,130,875,904 | 19.131 | 64.49 | retained evaluation payload, **sole copy** |
| `training/` | 10,526,916,608 | 10.527 | 35.49 | |
| ├ seed 20260815 | 4,654,235,648 | 4.654 | 15.69 | |
| │ ├ `stage_controllers/` | 4,628,283,392 | 4.628 | 15.60 | intermediate solver state |
| │ ├ `checkpoints/` (15 × ~1.06 MB) | 17,825,792 | 0.018 | 0.06 | **P0 + scored artifact** |
| │ ├ `retained/` (28 CAS manifests) | 5,767,168 | 0.006 | 0.02 | **verdict-critical** |
| │ └ `evaluations/` (14 JSON) | 1,966,080 | 0.002 | 0.01 | **verdict-critical** |
| └ seed 20260816 | 5,870,321,664 | 5.870 | 19.79 | |
| &nbsp;&nbsp;├ `stage_04_from_epoch_0000/` | 4,630,118,400 | 4.630 | 15.61 | intermediate solver state |
| &nbsp;&nbsp;├ `…chunk60_oom_retained_20260827/` | 1,214,119,936 | 1.214 | 4.09 | r1/r2 OOM dir, **already carries RETENTION_CERT** |
| &nbsp;&nbsp;└ checkpoints + retained + evaluations | 25,753,600 | 0.026 | 0.09 | **verdict-critical + P0** |
| `fire_orders/` + `launch_requests/` | 4,194,304 | 0.004 | 0.01 | governance |
| top-level receipts | ~334,000 | — | — | governance |

Two structural facts the table makes visible.

**(a) The training tree is 99.4% intermediate solver state.** Inside
`stage_controllers/stage_04_from_epoch_0000/` (4.628 GB, seed 1):
`quantization_race/` 2.929 GB · `controller_baseline_uniform4/` 0.976 GB · `gradient_chunks/`
0.484 GB · `selection_chunks/` 0.119 GB · `selective_cell_mask.u8` 0.118 GB ·
`gradient_accumulator.pt` 257,773 B · `STAGE_CONTROLLER_RESULT.json` 597,626 B.
The checkpoints — the P0-mandatory, actually-scored artifact — are **1.06 MB each**.

**(b) The CAS holds raw camera frames.** Object census: 1,740 objects at exactly
3,052,008 B = `874 × 1164 × 3` (5.310 GB, 28.1%), plus 561 scorer-field objects (13.556 GB,
71.9%), dominated by 56 × 117,964,999 B and 28 × 235,929,799 B. `117,964,800 = 600 × 512 × 384`
— the full argmax cell field. The 28 `38,847 B` objects are the W96 packets themselves.

The CAS is referenced by exactly **28 manifests, all inside this tree** (14 per seed). Nothing
outside the w96a tree references it. The source payload was compacted into it
(`"compacted": true`) and the source removed, so the CAS is the **sole copy**.

## 3. Minimum sufficient retention — the consumer trace

The consumer is the S1E composed-delta instrument, `tools/s1a_off_floor_adjudicator.py` (687
lines). Traced at source, it opens exactly two path families:

- `_EVALUATION_RE = re.compile(r"epoch_(\d{4})_n60\.json\Z")` (ln 42)
- `(seed_root / "stage_controllers").glob("stage_*_from_epoch_*/STAGE_CONTROLLER_RESULT.json")`
  (ln 165, ln 498)

It reads the JSON summaries. It does **not** open `quantization_race/`,
`controller_baseline_uniform4/`, `gradient_chunks/`, `selection_chunks/`,
`selective_cell_mask.u8`, or any CAS object. The downstream chain — byte-close → seal → T4 —
consumes one checkpoint per arm plus its config, not the solver intermediates.

Measured, both seeds:

| class | bytes | note |
|---|---:|---|
| `TRAIN_RESULT.json` × 2 | 256,927 | cited by both verdict memos |
| `evaluations/epoch_*_n60.json` × 28 | 870,909 | the adjudicator's primary input |
| `stage_*/STAGE_CONTROLLER_RESULT.json` × 2 | 1,195,284 | the adjudicator's second input |
| `stage_*/CONTROLLER_BINDING.json` × 2 | 6,510 | config identity |
| `retained/evaluations/**/*.json` × 56 | 2,624,563 | CAS manifests + evaluation results |
| `checkpoints/*.pt` × 30 (**all**) | 32,244,346 | P0 per-stage + the scored artifact |
| **VERDICT-CRITICAL TOTAL** | **37,198,539** | **0.0372 GB — 0.125% of the tree** |
| NON-VERDICT-CRITICAL | 29,625,705,781 | 29.626 GB — 99.875% |

Answering the mandate's three-way split:

- **(a) resume-critical while live** — the full `stage_controllers/` working set, ~4.6 GB per
  seed. Genuinely required *during* the run. Zero consumers after it.
- **(b) measurement artifacts needed for the verdict** — **37,198,539 B**, above. Keeping only
  the `stage_end` checkpoint instead of all 15 per seed would cut this to **7,135,991 B**, but
  I do **not** recommend that: the per-stage checkpoints are the P0 crash-insurance *and* the
  per-stage A/B measurement surface, and at 32 MB they are not worth optimizing.
- **(c) rebuildable bulk that certify-or-block permits cold-MOVING** — 29,625,705,781 B.

**The retention law, stated transferably:** for a WD3/S1A-class window, verdict-critical
retention is ~18.6 MB per seed and is entirely JSON plus 1.06 MB checkpoints. Everything else is
working set. A rolling policy that cold-moves each `stage_controllers/stage_*/` subtree at stage
end, and CAS-compacts evaluations as they are produced, would have run this experiment inside
**well under 1 GB** of steady-state footprint above the live working set.

## 4. Reshaping — priced, and moot for w96b

The arm already ran, so these are priced for the *next* window of this class, not for a re-fire.

| option | measured effect | sound? |
|---|---|---|
| **One seed instead of two** | halves demand to ~14.8 GB | **NO — and this is the important one.** The OFF baseline is two-seed, but the aligned *closure* needed both seeds: seed 1 alone returned 1.186×, inside the closure zone, and the charter still required seed 2 before closing. A one-seed aligned row against a two-seed OFF baseline re-opens #1251's single-draw problem exactly. The measured seed spread vindicates the caution and its cheapness: seg endpoints 3.98e-4 / 4.01e-4, a 0.8% spread. Two seeds cost 15 GB and bought a formulation-scope closure instead of an instance-scope one. That was the right purchase. |
| **Rolling retention (keep last-K stage subtrees)** | removes ~9.2 GB of the 10.5 GB training total | **YES.** Nothing downstream reads the solver intermediates. This is the single highest-value change and it is pure win. |
| **EMA-shadow-only persistence** | ~0 | **NO EFFECT HERE.** Checkpoints are 1.06 MB; whatever they carry, they are 0.1% of the tree. I did not open them to determine whether both live and EMA weights are stored — **ABSENT**, and not worth measuring at this size. |
| **Telemetry / frame-cache exclusion** | up to 5.31 GB of camera frames in the CAS | **PARTIAL.** These are materialized payload under ALWAYS-KEEP-THE-PAYLOAD, not telemetry. They may be moved, not excluded. |
| **CAS-compact the `stage_controllers` bulk** | unmeasured, likely large | **YES, and this is an apparatus gap.** `quantization_race/uniform2/` holds `receiver_pairs.rgb.u8` (366,240,960 B) and `scorer_outputs.npz` (365,694,134 B) — the *same kinds* the CAS chunks at frame and ZIP-member boundaries. `content_addressed_retention` was applied to `retained/evaluations/` and **never to `stage_controllers/`**. Confirmed: 28 manifests exist, all under `retained/evaluations/`, none under `stage_controllers/`. |

### Compressibility, measured (streamed to `/dev/null`; zero bytes written)

zstd level 3, sampled per size class, weighted by the object census:

| class | bytes | ratio | compressed |
|---|---:|---:|---:|
| camera frames 3,052,008 B × 1,740 | 5,310,493,920 | 4.4240× | 1,200,370,320 |
| scorer field 117,964,999 B × 56 | 6,606,039,944 | 1.1703× | 5,644,740,616 |
| scorer field 235,929,799 B × 28 | 6,606,034,372 | 1.1707× | 5,642,807,185 |
| field 11,796,678 B × 28 | 330,306,984 | 154.4917× | 2,138,024 |
| remainder (assumed 1.0×) | 13,700,155 | — | 13,700,155 |
| **CAS total** | **18,866,575,375** | **1.5089×** | **12,503,756,301** |

In-place zstd-3 on the CAS would free **6,362,819,073 B (6.36 GB)**. That is well below SR3's
measured 2.557×/2.568× on its trees, because the two scorer-field classes are 70% of the mass
and barely compress at level 3. Higher zstd levels and byte-shuffle filters are **unmeasured**.

## 5. Leg 4 — supply-side certification

MAIN's relay proposed `ddm_wd2_width_distillation` (32 GB) and `ddm_w96a_aligned_window` (28 GB)
as certification candidates, flagged by **arm status**, and correctly warned that a superseded
arm does not imply superseded artifacts. That warning was the right instinct. Both candidates
fail the scratch test, for different reasons.

### `ddm_wd2_width_distillation` — **KEEP_CITED**

Not scratch. The **live successor consumes it.**
`.omx/research/ddm_wd3_scorer_aware_width_distillation_20260815.json` — wd3 being the 58 GB LIVE
row MAIN told me not to touch — names four paths *inside* the wd2 tree as inputs. I verified all
four are present:

| cited path (under `/Volumes/APDataStore/pact/ddm_wd2_width_distillation/`) | bytes | status |
|---|---:|---|
| `teacher_cache_e480b/retained/teacher/teacher_master_camera.rgb.u8` | 1,831,204,800 | PRESENT |
| `primary_flattened_d4_w64/checkpoints/flattened_d4_w64/distill_qat_stage_end_epoch_0060.pt` | 583,929 | PRESENT |
| `primary_flattened_d4_w64/TRAIN_RESULT.json` | 16,234 | PRESENT |
| `…/advisory_n600_cpu/contest_auth_eval.json` | 25,848 | PRESENT |

There are **116 SSD-path citations** of this tree across the corpus. Two further live elements:
`python_shim_bin/` is the exec-wrapper shim recorded as a standing **asset** in
`ddm_hv1_pointer_move_and_wd2_advisory_chain_20260815.md` ("CURE + ASSET"), and
`ddm_av2_fresh_eyes_…` anchors a withdrawn-claim correction to `TRAIN_RESULT.json` sha
`c4260cf03e…`.

One trap avoided: most of the 51 corpus hits point at the **repo** directory
`.omx/research/ddm_wd2_width_distillation_build_20260815/` (small JSONs, in git), not the SSD
tree. I separated them. The SSD tree is still cited 116 times, so the conclusion holds on the
narrower evidence.

I checked the obvious counter-hypothesis: `ddm_wd4_warm_lineage_width` warm-starts from **fx5**,
not wd2 — its manifest is `retained/source_fx5/fx5_semantic_38_tensor_manifest.json`. wd4 cites
wd2 only as prior measurement. That does not rescue a scratch classification, because wd3 does.

### `ddm_w96a_aligned_window` — **KEEP_CITED (37,198,539 B) + MOVE_ONLY (29,625,705,781 B)**

MAIN's flag reads "#1298 honest-blocked". That is stale twice over. This tree is not a blocked
arm's leftover — it is the **completed output of w96b's two-seed run**, and it holds the retained
payload behind a MEASURED family closure. The two verdict memos cite named paths and SHAs inside
it: stage checkpoints `856707a852225e35…` and `8a51328640914585…`, both `TRAIN_RESULT.json`, both
`evaluations/epoch_0065_n60.json`, and the retain-MOVED chunk-60 dir with its RETENTION_CERT.

For the remaining 29.63 GB I **decline to certify delete**, and I want to be explicit about why,
because the certify-or-block rule would arguably permit it:

- It is materialized evaluation payload — camera frames and scorer fields — under
  ALWAYS-KEEP-THE-PAYLOAD, and the CAS is the **sole copy**.
- It is "rebuildable" only by re-running ~2 h of Metal training per seed. That is a real rebuild
  command, not a trivial cache or build product. The rule reserves destructive delete for
  "trivial caches/build products or explicitly certified rebuildable scratch"; a 4-hour GPU
  rebuild of a sole-copy payload is neither.
- The `chunk60_oom_retained` subtree already carries a RETENTION_CERT from a prior arm. Deleting
  what another arm certified for *retention* would be a custody regression.

So: **MOVE_ONLY**. The certificate below is producible in full.

### The honest consequence for MAIN's hope

MAIN wrote: *"If leg 4 shows 60 GB is genuinely certifiable, the operator ask may dissolve
entirely."* It does not show that. **Zero of the 60 GB is certified-rebuildable scratch.**
wd2 is cited by the live successor; w96a is sole-copy retained payload. The ask dissolves
anyway — but for the reason rb1 gave at 01:43 today (no live consumer), not because the bytes
turned out to be scratch.

What w96a's closure *does* change on the supply side is real and new: SR3 (08-26) listed
`ddm_w96a_aligned_window` among *"2 exact stores protected without mutation"*. It was protected
because w96b was about to fire into it. **That protection reason expired at family closure on
08-27.** The tree now meets SR3's own "terminal, superseded owner" criterion — the same test
that qualified SA1 and B2E for reclaim. It is newly *eligible* supply, at MOVE_ONLY, if a live
consumer ever appears.

### Feasibility, if a consumer ever appears

Per rb1 §5, archives are written **inside** the tree, so peak usage is originals + archive
against 7.63 GiB free and a 2 GiB abort floor. Whole-CAS zstd-3 needs 12.50 GB of headroom to
write. **It does not fit today.** Per-seed `stage_controllers/` subtrees are the only granularity
that plausibly fits, and their compressibility is **unmeasured**. I did not attempt it: no
consumer, and MAIN/operator executes.

## 6. Verdict

**`w96b` is not fireable in 7.5 GiB, and the question is void.** It already fired, in 29.66 GB
of realized retention, and its family is CLOSED at formulation scope by its own chartered fork
rule. The `≥5×` gate failed at both seeds (1.186× and 1.5796×). The dominant failure term is
**pose**, ~80% of the composed delta at both seeds, with d_pose 185–204× gb1's 6.37e-6.

**No operator storage ask stands.** I concur with `ddm_rb1_pose_arithmetic_closure_and_storage_no_consumer_20260831.md`
on independent evidence. My ~15 GB escalation was wrong on both halves: the demand figure was a
free-space figure, and the arm it would have unblocked was already dead.

The causal chain, closed:

1. **08-26** SR3 reclaimed 32.75 GiB → AP free 44.8 GiB → w96b's storage trigger went **GREEN**.
2. **08-27** w96b fired both seeds, consumed 29.66 GB, and the family **CLOSED**.
3. **08-27** bs4y retained ~60 GB of stage_20 solve payloads.
4. **08-31** AP sits at 7.1–7.6 GiB free.

Current AP pressure is the **result** of w96b having run, not the cause of it not running.

Reactivation is unchanged and belongs to w96b, not to me: an aligned-family trained renderer
re-enters only as a **pose-carrying object change** — measured d_pose within ~4× of 6.37e-6 at
n≥60, or composed_delta within +0.028 of renderer break-even. Not on seg, seeds, or width.

## 7. Fire order

1. **NO-OP — storage ask.** Owner: MAIN. **Do not put a storage question to the operator.**
   Superseded by the 08-31 rb1 adjudication and independently confirmed here.
2. **CORRECT THE LEG-4 CANDIDATE LIST.** Owner: MAIN. Reclassify
   `ddm_wd2_width_distillation` → **KEEP_CITED** (live successor wd3 consumes its 1.83 GB
   teacher cache and its epoch-0060 checkpoint; 116 SSD-path citations). Reclassify
   `ddm_w96a_aligned_window` → **KEEP_CITED 37,198,539 B + MOVE_ONLY 29,625,705,781 B**, and
   record that its SR3 protection expired at family closure so it is newly eligible supply.
   Neither is certified-rebuildable scratch. Fire trigger: immediate, documentation only.
3. **QUEUED — CAS-compact `stage_controllers`.** Owner: a retention successor. The apparatus
   gap: `content_addressed_retention` is wired to `retained/evaluations/` only. Extending it to
   `stage_controllers/*/quantization_race|controller_baseline_*|gradient_chunks` would have
   covered ~9.2 GB of the same payload kinds this tree already chunks. Fire trigger: the next
   WD3/S1A-class window is chartered. Not urgent — no live window.
4. **QUEUED — rolling stage retention.** Owner: same. Cold-move each `stage_controllers/stage_*/`
   subtree at stage end. Measured saving on this tree: ~9.2 GB of 10.5 GB training footprint,
   with zero verdict impact — the adjudicator reads only `STAGE_CONTROLLER_RESULT.json`.
   Fire trigger: same.
5. **HELD — any w96a reclaim.** Owner: MAIN/operator. Certificate fields below are producible.
   Fire trigger: a live consumer appears **and** free space admits the peak. Neither holds today.

### Certificate skeleton (MOVE_ONLY, w96a non-verdict bulk)

Producible in full on demand; I did not execute it.

```
original_path : /Volumes/APDataStore/pact/ddm_w96a_aligned_window/{content_addressed_store,
                training/aligned_seed_2026081{5,6}/W96_flattened/stage_controllers}
bytes         : 29,625,705,781  (CAS 19,130,875,904 + stage_controllers 10,494,830,272)
sha256        : per-object, already durable — 2,301 CAS objects are SHA-256-NAMED; the 28
                CAS_RETENTION_MANIFEST.json files carry path+bytes+sha256 for all 3,164
                logical files. stage_controllers needs a fresh manifest (not yet produced).
rebuild_cmd   : re-run both sealed windows — fire orders aligned_seed_20260815_authorized_20260827
                and aligned_seed_20260816_chunk30_20260827 (config_sha256
                13163dc82514aaad1f74c898e4ae00d8d7968e9e7d3b8970069a381157041915), ~2 h Metal/seed
why_movable   : family CLOSED 08-27 at formulation scope; no consumer reads these paths; SR3's
                08-26 protection reason expired at closure
why_NOT_delete: sole copy of materialized evaluation payload (ALWAYS-KEEP-THE-PAYLOAD); rebuild
                is 4 h of Metal, not a trivial cache; chunk60 subtree already carries a
                prior arm's RETENTION_CERT
cold_dest     : ABSENT — no sanctioned tier has 29.63 GB free (AP 7.1 GiB, Vertigo 8.3 GiB).
                BLOCKER, per certify-or-block: proof of destination missing, so keep the bytes.
keep_paths    : the 37,198,539 B verdict-critical set enumerated in §3
```

## 8. Denominator

**Components enumerated: 11. Measured: 11. Sub-quantities ABSENT: 5.**

Measured: (1) CAS allocated size; (2) CAS object census by size class; (3) CAS dedup factor from
all 28 manifests; (4) CAS zstd-3 compressibility, 4 size classes, 6 objects; (5) seed-15
subtree breakdown; (6) seed-16 subtree breakdown incl. the chunk-60 OOM dir; (7) verdict-critical
byte set, both seeds, by consumer trace; (8) adjudicator consumer paths at source; (9) CAS
external-reference check; (10) wd2 cited-path presence, 4 of 4; (11) SSD-path vs repo-path
citation split for both candidate trees.

ABSENT, named rather than filled:
- Full `du` over all 256 APDataStore trees — my background job died `rc=144` (harness reaper) on
  ExFAT. MAIN's relay supplied the census; I did not re-derive it.
- zstd levels above 3, and byte-shuffle filters, on the two scorer-field classes.
- `stage_controllers` compressibility — the 10.5 GB I recommend moving first.
- `ddm_wd2_width_distillation` internal size breakdown (I established citation, not composition).
- Whether the 1.06 MB checkpoints store live weights, EMA shadow, or both. Immaterial at 0.1%.

## 9. Prior-law prediction — outcome

I predicted: demand dominated by rebuildable bulk, verdict-critical under 5 GB, arm fireable in
current free space with rolling retention. Falsifier: demand irreducibly verdict-critical above
7.5 GiB, in which case the operator ask stands.

**The falsifier did not fire.** Verdict-critical is 37,198,539 B — 0.125% of the tree, 134×
below my own 5 GB bound. With rolling retention this experiment fits in current free space with
room to spare.

But the prediction was **answering a dead question**, and that is the more useful correction.
I framed this as "can we make the arm fit?" when the arm had already run four days earlier and
the ask had been withdrawn six hours before I was spawned. The measurement is sound and the
retention law is transferable. The framing was stale, and no amount of careful measurement
inside a stale frame would have surfaced that — only reading the successor did.

## GESTALT-DELTA

Before this unit the working story was: "R+P's last unmeasured member is storage-blocked; buy
~15 GB." Three things change.

1. **The member is not unmeasured.** It is measured and CLOSED, with pose named as the dominant
   failure term at ~80% of the composed delta at both seeds. The aligned seg law was *vindicated*
   ~2.03× and seed-robust to 0.8% — the family died on the axis nobody was watching, which is the
   same shape as `[[m110]]`'s pose absolute budget and `#1222`'s "PoseNet scores the FRAMES".
2. **Storage was never the campaign's binding constraint here; it was a lagging indicator.**
   AP hit 100% *because* the science ran. Reading disk pressure as a blocker to be purchased away
   inverted cause and effect.
3. **A retention law is now measured and transferable:** for this trainer class, 0.125% of a run
   tree is verdict-critical, and the consumer reads JSON. The apparatus already contains the cure
   (`content_addressed_retention`) and simply is not wired to the 35% of the tree that needs it
   most. That is a cheap, real fix for the next window.

Method note for my own class of error: this arm was one `ls` away from the truth at minute one.
The memo titled `…_verdict_and_family_closure_` was sitting next to the charter I was handed.
`[[a_reopen_must_check_whether_the_successor_already_fired_20260829]]` names this exactly, and it
fired again anyway — because the check was carried in a mandate's prose instead of at the spawn
site. `[[m122]]`'s cure is the right one: recall belongs to the apparatus, not to volition.

## RECALL EVIDENCE

Searched `.omx/research/` by content for `w96`, `w96a`, `w96b`, `aligned_window`, `pk4`, `sr2`,
`sr3`, `reclaim`, `route 3`, `rb1`, `wd2`, `wd3`, `wd4`, `warm lineage`, `CAS`, `retention`,
`certify`; the charters and `arm_final_messages` subtrees; `.omx/state/canonical_task_status.jsonl`,
`durable_daemons.json`, `codex_arm_queue.next_if_resumed.jsonl`; `tools/`, `src/tac/`,
`experiments/` for retention and adjudicator source; and the live APDataStore trees.

Findings that changed the work:

- `ddm_w96b_seed20260815_aligned_verdict_20260827.md` + `…seed20260816_aligned_verdict_and_family_closure_20260827.md`
  — the arm already fired and closed. This inverted the entire mandate.
- `ddm_rb1_pose_arithmetic_closure_and_storage_no_consumer_20260831.md` (today, 01:43) — the ask
  was already withdrawn on all three legs. My escalation was superseded before I began.
- `ddm_rb1_route3_arithmetic_closure_20260827.md` — route 3 closed by arithmetic on 08-27,
  exactly as the w96b closure memo predicted it would.
- `ddm_sr3_ap_certify_compress_reclaim_20260826.md` — supplied the measured 2.557×/2.568×
  compression references, the "terminal, superseded owner" test I applied to w96a, and the record
  that w96a was *protected* on 08-26 for a reason that has since expired.
- `ddm_wd3_scorer_aware_width_distillation_20260815.json` — the four wd2-internal paths the live
  successor consumes. This is what turned MAIN's wd2 scratch hypothesis into KEEP_CITED.
- `ddm_wd4_warm_lineage_width_20260821.md` — wd4 warm-starts from fx5, not wd2; the obvious
  counter-hypothesis, checked and rejected on its own terms.
- `#1301`, `#1298`, `#1165`, `#1024`, `#1336` are **ABSENT** from
  `.omx/state/canonical_task_status.jsonl`. They are harness TaskList ids, not repo-ledger ids —
  `[[m89]]`. I cite content throughout, never bare ids.

## Retained receipts

| path | bytes | SHA-256 | disposition |
|---|---:|---|---|
| `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/W96B_BUILD_AND_STORAGE_RECEIPT.json` | 33,416 | `6fac1bd8c8c40f9fc97df57b341b0b99c010105ecbdf12c2495304e7491cd42b` | KEEP; the demand receipt this memo reconciles |
| `…/STORAGE_PREFLIGHT_BLOCKER.json` | 1,870 | `4e62de066d3304e970a20002aa08400789015f9d26d69f143d82f4227f3e1056` | KEEP; the 08-26 blocker, now historical |
| `…/SEALED_FIRE_ORDER_W96B.json` | 2,723 | `92d9047563a3c9da7c1250a97b475f4e14c5c93ea2eb2994151c13200301b8f2` | KEEP; the order that fired |
| `…/off_baseline_s1e_rerun.json` | 296,698 | `3037d264f097cd1b239cd96fc2302f5d812e0f3384eea7813fdc5cb074b60b18` | KEEP; OFF replay |

This arm wrote no artifact to either SSD. Nothing was deleted, moved, or compressed. All zstd
measurements streamed to `/dev/null`. No scorer, Metal, training, Modal, or contest evaluation
ran. `upstream/` was not touched.

**[contest-CUDA T4 n600] own-vehicle frontier: LB1 — S=0.14803010583079396, archive=180,083 B,
d_seg=0.00020139, d_pose=6.37e-6,
SHA-256=5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9 — UNMOVED by ddm_w96s.**
