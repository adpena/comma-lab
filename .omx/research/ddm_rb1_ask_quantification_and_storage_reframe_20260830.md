# ddm_rb1 — the ASK quantified, and the storage blocker re-aimed at the wrong volume

`axis: [macOS-CPU scorer-free arithmetic + source read]` · `score_claim: false` · `promotable: false`
`verdict_scope: none — this memo states a BAR and corrects a routing error; it closes nothing.`
Date: 2026-08-30 · Owner: MAIN · Consumers: #1304 (storage-gated triple fire) · #1165 (Vertigo
reclaim round 2) · the rb1 sealed configs.

APPEND-ONLY: this supersedes nothing in
`.omx/research/ddm_rb1_born_small_renderer_build_20260826.md`. rb1's own numbers are reproduced
here and were all correct; what was missing is an arithmetic rb1 honestly declined to do
(its §111: *"no inherited distortion or score projection was admitted"*).

## 1. The routing error I made, corrected before it cost anything

I spent most of a window aiming a certified cold-MOVE at **pk4 (106.09 GiB on Vertigo)** to
unblock rb1. Verifying the premise before putting a question to the operator inverted the target:

- rb1's fire trigger (its §95) is *"SR3 is green with at least 60,380,026,816 B free"*.
- SR3 (`ddm_sr3_ap_certify_compress_reclaim_20260826.md`) is an **APDataStore** reclaim.
- `ddm_vr1` line 118 independently confirms the binding constraint is AP: cb1's 30.71 GiB move
  was refused because *"moving it would drop APDataStore below the 25 GiB floor."*

**Moving pk4 off Vertigo frees the wrong volume.** Measured now:

| | bytes | GiB |
|---|---:|---:|
| AP free | 8,210,481,152 | 7.65 |
| rb1 requires | 60,380,026,816 | 56.24 |
| **shortfall** | **52,169,545,664** | **48.59** |

Also corrected: the "36.9 GiB short" figure I was carrying from #1165 is stale. rb1's own
receipt says the retained preflight observed 4,624,875,520 B free against a
**55,755,151,296 B shortfall** at the time it sealed.

## 2. The ask nobody had written down

rb1's byte gate passed with room (its own §9 table, D56 rung):

```
BS3 body 101,150 + Brotli-q11 renderer 17,845 + RB1A/ZIP framing 180 = 119,175 B
credit vs the 137,986 B sub-0.12 cap                                 =  18,811 B
```

So rb1 is **byte-feasible**. Its distortion has never been measured. Deriving the requirement
from measured quantities only:

```
rate      = 25 · 119,175 / 37,545,489            = 0.079354
budget    = 0.12 − 0.079354                      = 0.040646
pose held at lb1's measured 0.007981227975693965
seg budget                                       = 0.032665
required d_seg                                   = 0.00032665
```

The nearest measured neighbour — a born-small field carried by an **inherited** renderer — is
`ddm_bz2d` at **d_seg = 0.01299522**. Therefore:

> **rb1's trained renderer must cut d_seg 39.8× — correct 97.49% of the argmax errors its own
> frozen token field causes.**

And `ddm_gf1` measured what that correction information costs: **0.2909 coded B per correction**
on this generator's clustered residual, × 1,325,033 mismatches = **385,452 B**. rb1's renderer
stream is **17,845 B** — a **21.6× information-budget gap**.

⚠ This is a PRIOR, not a theorem, and the direction of every caveat is stated:
- 0.2909 B/correction was measured under **generic LZ** (lzma2/brotli/zlib). A convolutional
  renderer is a far stronger model on **clustered** errors, and gf1 §3a measured these errors
  *are* clustered (tile16_time beats frame_raster by 15.5%). A CNN beating LZ by 20× on
  structured residual is ordinary neural-compression behaviour, not a miracle.
- d_seg 0.01299522 is a **lineage** transfer, not byte-identity: bz2d measured field
  `968ffca2…`; rb1 trains on field `2884c570…` (the HG1 generated field shared with
  bs2/bo2/bs3/rd2). gf1 measured this generator's capacity ceiling is target-independent to
  0.04%, which licenses the transfer at family scope — stated, not assumed.

## 3. The decisive code fact: rb1 is NOT a distillation ceiling

I expected to find that rb1's student is trained toward its **teacher's** argmax — in which case
a student could never beat its teacher, rb1's ceiling would be bz2d's 99.68×, and the whole
60 GB storage question would be moot. **That is not what the code does.**

`experiments/ddm_wd3_scorer_aware_width_distillation.py` carries **three** argmax fields, not two
(`:908-917`): `student_argmax`, `teacher_argmax`, `original_argmax`. In `cell_edge_telemetry`
(`:930-946`) the **`original_argmax` is the target** — it is literally named `target` and its
values `expected`. The teacher enters only as *constraints*: `DualState` (`:968-972`) carries
`margin`, `teacher_kl`, `decode`, `teacher_pose` as nonnegative duals.

> **The renderer optimizes toward GT with the teacher as a KL/pose constraint. It is a genuine
> correction channel, and it CAN in principle beat the object bz2d measured.**

So rb1 survives both bz2d (99.68×) and bo2 (209×) for exactly the reason its own §102
pre-registered — those are inherited-renderer rows, and rb1's distinct mechanism is training the
renderer on the changed object. That defence is sound and I could not break it.

## 4. What this changes for the fire decision

rb1 is a real, unmeasured mechanism with a 21.6× prior against it, and the route table is
otherwise EMPTY. That makes it worth measuring — but **not at four configs.**

- Four sealed configs: 4 × 12,489,721,856 B + 1,831,204,800 B teacher master ≈ **60.38 GB**
  (48.59 GiB short).
- **One** config + the shared teacher master ≈ **14.3 GB** (≈6.1 GB short of AP's current 7.65 GiB).

One config answers rb1's own question — *does a GT-targeted trained renderer attenuate
token→argmax error at all, and at what slope?* The four-config matrix prices variants of an
effect whose existence is unmeasured. The measurement-first discipline says buy the existence
test first, and it is **~4.2× cheaper in storage**.

## 5. Owed, and to whom

1. **AP-side reclaim, not Vertigo.** The permission-free path is SR3's own precedent —
   certify + **compress in place** (AP→AP, deterministic zstd, round-trip verified), which took
   AP from 12.05 → 44.8 GiB free on 08-26 without touching the local tier. Post-SR3 lanes carry
   zero `*.tar.zst`: `ddm_bs3` 65.0 GiB, `ddm_qbflow` 62.1 GiB (both arms landed),
   `ddm_wd3` 57.9 GiB (**do not touch — #1273 is open**). Each needs its own closure evidence
   before compression; "the arm landed" is not "the retention is certified rebuildable".
2. **The local tier stays an operator decision.** `tools/vertigo_certify_move.py` now refuses a
   local destination without `--allow-local-tier '<rationale>'` (landed `a11005cbb9`, this
   window). I built that guard; I am not going to exercise it on my own authority hours later.
   Note it is also **Vertigo-source-only** by invariant, so an AP→local move needs a different
   tool — a real gap, not a workaround.
3. **The `_manifests/` count is 0.** `/Volumes/APDataStore/pact/_manifests/` holds no entries and
   `cold_store/` holds one. So AP's ~322 GiB of `cold_store*` directories are **not certified
   through this tool's ledger**. My earlier phrasing ("already-certified cold store") was an
   assumption; it is withdrawn. Each candidate needs its own certification check.

## 6. Honesty

No score is claimed. §2's arithmetic is derived from measured inputs (rb1's own byte table,
lb1's pose contribution, bz2d's d_seg, gf1's correction price); §3 is a source read with file and
line numbers; §1's shortfall is `df` at the time of writing. Nothing here measures rb1's actual
distortion — that measurement does not exist, which is precisely the finding.
