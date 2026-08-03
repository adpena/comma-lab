# ddm_gd4 — `grid_downsample=32`: the gate is OPEN and the run is SEALED; two blockers gd3/mt1 did not name are now MEASURED, and one of them is load-bearing for the row

**Arm:** `ddm_gd4`, acting on `ddm_gd3`'s recommendation (`.omx/research/ddm_gd3_grid_downsample_gate_20260802.md`, commit `db3abc5b4a`).
**Axis:** `[macOS-CPU/MLX advisory — real trainer, real n600 gt cache, real DSL compile, real governed launcher]`.
`score_claim=false`, `promotion_eligible=false`, `pointer_moved=false`. Repo HEAD at run: `db3abc5b4a`.

---

## §0 POINTER HONESTY + THE ANSWER

**Exact contest pointer UNMOVED at `0.1910828242 [contest-CPU]`. No candidate gated. No archive
built. No paid dispatch. No heavy run fired — MAIN fires; this unit seals.**

Anchoring (see §0.1 for the gap reconciliation): LIVE BEST = `dc1_fold`, 360,309 B,
seg 0.4311790 / pose 0.2272835 / rate 0.2399150 → **S 0.8983775**. Bar = PR130 **0.1721417**
@ 191,052 B. **Gap 0.7262358**; 1% of gap = 10,907 B = 0.0072624 S.

| deliverable | state |
|---|---|
| **G1 gate opened honestly** | **DONE + POSITIVE CONTROL.** ds=32 builds a genuine 7-conv decoder; `up4` is registered trainable, carries **nonzero gradient**, and has a **causal render effect (absmax 79.33 vs the ds=16 `up3` control's 76.08)**. It is not the inert-lever class. Three DSL/argparse menus widened; a regression test **fails if `up4` is ever gradient-starved**. |
| **G1 defect found + fixed** | **A ds=16 checkpoint loaded into a ds=32 model WITHOUT RAISING.** `mlx.nn.Module.update` silently assigns wrong-shaped arrays. Fixed fail-closed; 672 real ds=16 checkpoints on disk made this a live hazard. |
| **NEW blocker #1 (load-bearing)** | **`token_cell_mask` fail-closes at ds=32 — and the row's size DEPENDS on it.** Maskless ds=32 costs **189,340 B**, which **TRIPS gd3's own falsifier #2 (≥187,000 B)** and collapses the row from 23.7% → **15.7%** of the gap. Three ds=32 masks derived, priced and custodied. |
| **NEW blocker #2 (inert lever)** | The trainer REFUSED my incumbent-faithful ticket: `--margin-weighted-loss on` is **inert** under the form the run occupies. MEASURED by the guard: **the entire b4s/r1c burn carried this flag inert for 100% of its epochs.** |
| **G2 seal** | **DONE.** DSL-compiled ticket `d5ba6b30f1df152e…`; governed launcher **dry-run: ALL GATES PASS**; per-stage checkpoints **observed being written**, not merely read in the source. Peak RSS **CERTIFIED 15.014 GiB** at the true config — **1.17× the D16 anchor the launcher projects from**, refuting my own mid-flight projection (§4.3). |
| **G3 pose re-solve** | **BUDGETED into the row** (§5), with the live executor named and the charter's `#850` hint corrected: that GN path is **dead code on this vehicle**. |
| **G4 falsifier** | **PRE-REGISTERED before the run** (§6), with the threshold restated against the arm actually sealed. |

### §0.1 The gap number — reconciled, as asked

gd3 quotes **0.7262365**; the charter quotes **0.7262358**. **I used 0.7262358.** Reason: it is the
value `tac.canonical_equations.gap_decomposition_against_floor_20260802` returns against
PR130 = 191,052 B, and na1's 2026-08-02 correction established 191,052 B (not 190,952 B) as the
floor that reproduces PR130's published 0.172141. gd3's figure differs by **7.0e-07** — 0.0001% of
the gap, ~0.01 B of archive. It changes no verdict here; I name it only so the two memos do not
silently disagree. All §4 percentages are against 0.7262358.

---

## §1 G1 — THE GATE IS OPEN, AND `up4` IS NOT INERT

### 1.1 What was widened (three menus, one row)

gd3 §6.1 said the argparse `choices` tuple was "the whole blocker". **It was one of three**, and the
other two only surface when you actually run the thing:

| surface | before | after |
|---|---|---|
| `experiments/train_tr1_partition_renderer_mlx.py:1692` argparse | `choices=(8, 16)` | `choices=(8, 16, 32)` |
| `TR1Config.grid_downsample` docstring | `D in {8, 16}` | `D in {8, 16, 32}` |
| `tac.witness_dsl.spec_tr1_renderer_20260728.lever_token_grid` (the **DSL SoT**) | `if downsample not in (8, 16): raise` | `(8, 16, 32)`; **D=12 still refused** (512/12 non-integer) |

Widening argparse alone would have been a **DSL bypass** — `#506` compile is fail-closed and the
governed launcher recompiles the ticket, so a hand-authored `--grid-downsample 32` would have been
REFUSED at G1 seal-freshness. Both menus had to move together.

### 1.2 The positive control (the charter's actual ask)

> *"A layer that exists but never receives gradient is the inert-lever class. Add a positive control
> that would FAIL if `up4` were inert."*

**MEASURED** (`test_ds32_up4_is_trainable_and_not_inert`, real MLX, three independent legs — the
inert case fails each one):

```
ds=16   conv0 up0 up1 up2 up3 head          6 convs   grid 24x32   render (1,384,512,3)
ds=32   conv0 up0 up1 up2 up3 up4 head      7 convs   grid 12x16   render (1,384,512,3)

leg 1  registered      : {s_up4, g_up4, b_up4} in trainable_parameters()   PASS
       (and `_bank` is NOT trainable — the lotto bank is rule-118-free PRNG expansion)
leg 2  gradient        : |grad| absmax > 0 for EVERY up{0..4} score AND bias   PASS
leg 3  causal on output: perturb ONLY b_up4 -> render delta absmax 79.33      PASS
       ds=16 control  : perturb ONLY b_up3 -> render delta absmax 76.08
```

Leg 3 is the one that matters: legs 1–2 can both pass on a layer whose output is discarded. The
ds=16 `up3` control gives the comparison scale — **up4 moves the render as hard as the incumbent's
last upsample does.** `ds=16` correspondingly has **no** `up4` attribute (no silent topology drift).

Confirmed in-vehicle at §3: a real ds=32 run trains, descends, and gates.

### 1.3 The MEASURED defect: `--resume-from` silently accepted a ds=16 checkpoint

This is the finding I did not expect and the one with the largest blast radius.

**Before the fix, MEASURED:**

```
save a ds=16 checkpoint -> load_checkpoint(it, ds32_model)
  raised?                              False          <-- no error at all
  meta.cfg.grid_downsample             16             <-- the mismatch IS recorded
  live model grid_downsample           32                  and never checked
  tokens_delta shape after load        (2, 24, 32, 4) <-- the ds=16 shape, in a
  tokens_delta shape expected          (2, 12, 16, 4)     12x16 renderer
  up4 left at init                     True
```

`mlx.nn.Module.update` assigns a **wrong-shaped** array without complaint. Worse, the resume block's
existing `backfilled` mechanism — written for legitimately newly-introduced params like
`head_relax_gain` — would have absorbed `s_up4/g_up4/b_up4` as "new params since the checkpoint" and
logged a **truthful-looking** `ema_backfilled_new_params` line. The run would look like a clean warm
start.

**Why this was live, not theoretical:** I scanned every `.npz` under the SSD tiers,
`experiments/results/` and `.omx/` — **16,326 files scanned, 1 unreadable, 672 real TR1 trainer
checkpoints found, ALL of them `grid_downsample=16`, `num_pairs=600`.** A multi-day ds=32 run whose
first crash-recovery action is `--resume-from` sits one tab-completion away from every one of them.

**The fix** (`assert_resume_geometry_compatible` + `ResumeGeometryMismatch`, called inside
`load_checkpoint`): **structural, not declarative** — it compares the checkpoint's `param::` tree
against the live `model.trainable_parameters()`, so it needs **no new argument** and protects
**every** caller (`ddm_b4r_endpoint_extras`, `ddm_pa1r_endpoint_verdict`, the trainer, the tests),
not just the trainer's resume block. It refuses on any shape conflict or orphaned checkpoint param,
and **returns** the model-only params so the legitimate backfill path is preserved.

Post-fix, the same probe:

```
ResumeGeometryMismatch: resume REFUSED — checkpoint geometry does not match the live model
  shape conflicts: ['tokens_base: ckpt(24,32,4) != model(12,16,4)',
                    'tokens_delta: ckpt(2,24,32,4) != model(2,12,16,4)']
```

**Sister guard, at the resume callsite:** `lotto_seed`. The fixed bank is regenerated from the seed
and is **not** checkpointed, so a seed change survives the structural check at **identical shapes**
while making every trained supermask index a different random bank. Fail-closed there too.

**Blast-radius check — I did not just assert the fix is safe, I ran it against production bytes:**
the **real** `b4s/window_03/checkpoints/intra_seg_trunk_tau_ep00945.npz` loads into a
current-source matched model with **no error**, `epoch == 945`, `params_new_since_checkpoint == []`,
render shape `(1,384,512,3)`. Six config fields have been *added* to `TR1Config` since that
checkpoint was written and none is a trainable param, so the guard is silent on them — correct.

---

## §2 NEW BLOCKER #1 — `token_cell_mask`, and why it decides the row's SIZE

### 2.1 The blocker

The incumbent burn config carries
`--token-cell-mask /Volumes/VertigoDataTier/pact/ddm_sg1_20260731/qa24_grid_keep_mask_50.npy`
— a `(24,32)` bool array. `build_module` fail-closes at ds=32:

```
ValueError: token_cell_mask shape (24, 32) != grid (12,16) at D=32; fail-closed (never-invent geometry)
```

Neither gd3 §6 nor mt1 §5 names this. It is a hard stop on the sealed config, and it cannot be
dissolved by dropping the flag — because of §2.3.

### 2.2 The mask is a row band, and the shipped token stream is DENSE

**MEASURED on the live `dc1_fold` archive** (real `decode_token_codes`, real smevr coder):

```
state/tokens.dr7t  346,478 B  ->  codes (600, 24, 32, 4) uint8  = 768 cells/frame  (DENSE)
cells constant across all 600 pairs: 384 / 768 = 0.5000
per-row constant: [32,32,32,32,32,28,22,18,10,2,1,0,0,0,0,0,0,0,1,14,32,32,32,32]
mask   per-row keep: [ 0, 0, 0, 0, 0, 4,10,14,22,30,31,32,32,32,32,32,32,32,31,18, 0, 0, 0, 0]
```

The two are **exact complements**, row for row. The mask is a contiguous **row band** (sky dropped
above, hood/MyCar dropped below) — reproducing gd3 §4.1's sky/road split independently.

Two consequences: (a) the shipped archive codes the **dense** 24×32 lattice including the 384
zeroed cells, so gd3's decimation of it is **apples-to-apples dense-lattice** and its ladder is
sound; (b) the mask's rate benefit is not cell removal at the coder — it is that **half the cells
carry no entropy**.

### 2.3 Pricing the fork — and the maskless arm trips gd3's own falsifier

Because (b) holds, the ds=32 rate estimate depends entirely on **how many ds=32 cells are active.**
I priced all four arms on the real shipped tensor through the real smevr coder. For each candidate
mask, cells the mask calls active but the field holds constant were filled with content copied from
a genuinely active cell (a clearly-labelled **synthetic** fill — the only way to price an
active-cell count that no trained ds=32 field yet exists to supply); cells the mask drops were
forced constant. Archive bytes add gd3's MEASURED renderer/framing overhead (`101,636 − 87,065`):

| ds=32 cell mask | active | row band | tokens B | archive B | ΔS_rate | % of gap |
|---|---:|---|---:|---:|---:|---:|
| `all` (aggressive) | 78 | rows 3–9 | 73,556 | 88,127 | −0.181235 | **25.0%** |
| `stride2` (**reproduces gd3's 87,065**) | 94 | rows 3–9 | 87,065 | 101,636 | −0.172240 | **23.7%** |
| **`any` (derivation-faithful) — RECOMMENDED** | 109 | rows 2–9 | 100,476 | 115,047 | −0.163310 | **22.5%** |
| **`none` (dense, i.e. just dropping the flag)** | 192 | rows 0–11 | 174,769 | **189,340** | −0.113841 | **15.7%** |

**The dense arm's 189,340 B is above gd3 §7 falsifier #2's 187,000 B threshold.** Dropping the flag
— the obvious way to make the fail-closed error go away — would have shipped a run that trips the
row's own pre-registered falsifier and shrinks it from 23.7% to 15.7% of the gap. **The mask is
load-bearing.**

**Recommendation: `any` (109 active).** Not because it is the largest number — it is the *smallest*
of the three masked arms — but because it is the only rule that preserves the mask's justification.
`cell_drop50` was derived as the keep-set carrying **99.61% of flip mass** (gr1); `any` keeps a
coarse cell iff **any** of its 2×2 fine children was kept, so that coverage guarantee survives
pooling. `stride2` reproduces gd3's byte count but keeps only fine child (0,0) — a phase-arbitrary
subsample that silently discards covered flip mass. **22.5% of gap on the honest mask beats 23.7%
on a phase artifact**, and it is still 3.2 pp clear of the false-flag alternative and well clear of
the falsifier.

**Custodied** (durable SSD, never `/tmp`) at `/Volumes/VertigoDataTier/pact/ddm_gd4_20260803/`:
`qa24_grid_keep_mask_50_D32_{any,stride2,all}.npy` + `ds32_cell_mask_provenance.json`
(source path + source sha256 + rule statement + the four measured arms).

---

## §3 NEW BLOCKER #2 — the incumbent burn carried an INERT lever, and the trainer caught it

I compiled the ds=32 ticket by mirroring the incumbent b4s window_01 config exactly. The trainer
**refused to start**:

```
REFUSED: --margin-weighted-loss on is INERT for seg form(s) ['tau_softplus'] that a run
         launched at --seg-form-start 'ce' will occupy.
  NOTE 'ce' is not a way out: the knee event (F2 midpoint fallback makes it unconditional)
  switches ce -> tau_softplus mid-run, so the weighting dies at the knee. This is MEASURED:
  b4s window_01..03 + r1c window_01 all launched margin_weighted_loss='on' and ran
  tau_softplus for 100% of their epochs.
```

The guard (`ddm_tp2`'s `assert_margin_weighted_loss_is_honored`) is right, and it states a
**measured fact about the entire incumbent burn**: every window declared this lever on and none of
them ever honored it. I dropped the lever — which the guard itself certifies is **byte-identical to
what those runs actually did**. Worth recording because "faithfully mirror the incumbent" was the
correct instinct and it still imported an inert flag; only *running* the config surfaced it.

---

## §4 G2 — THE SEAL

### 4.1 DSL compile (never hand-authored argv)

`tac.witness_dsl.spec_tr1_renderer_20260728.TR1RendererProgramV1` with 20 typed `Lever` factories;
`compile_trainer_argv()` runs `validate()` internally (never-invent-flags AST scan of the trainer's
own argparse).

* **window_01 ticket hash: `d5ba6b30f1df152e4e8478fa2cb575970bb2c557588f16e4c32ef943a5020b2b`**
* custody: `/Volumes/VertigoDataTier/pact/ddm_gd4_20260803/ticket_ds32_window_01.json`
* out-dir: `/Volumes/VertigoDataTier/pact/ddm_gd4_ds32_20260803/window_01`

**Diff vs the INCUMBENT `b4s/window_01` argv — exactly five entries, every one intended:**

| flag | incumbent | ds=32 ticket | why |
|---|---|---|---|
| `--grid-downsample` | `16` | **`32`** | **THE ROW** |
| `--token-cell-mask` | `…sg1…/qa24_grid_keep_mask_50.npy` | `…gd4…/qa24_grid_keep_mask_50_D32_any.npy` | §2 — required, and load-bearing |
| `--margin-weighted-loss` / `--margin-weight-temp` | `on` / `1.0` | *(absent)* | §3 — inert; removal is byte-identical to what the incumbent did |
| `--resume-from` | `…r1c/window_01/…tau_final.npz` | *(absent)* | **from scratch.** No ds=32 parent exists, and §1.3's guard would refuse a ds=16 one |
| `--out-dir` | b4s window_01 | gd4 ds32 window_01 | new run |

(`--renderer-head-mode rgb` is emitted explicitly where the incumbent relied on the default; same
value, same config identity.)

Everything else is byte-identical to the incumbent: `lotto`, `code_width 4`, `renderer_width 24`,
`token_quant_levels 16`, `token_ste round`, `shared_base`, `lotto_seed 118`, `w_seg 100.0`,
`class_weight_lane 1.3`, `margin_target 1.0`, `token_init_mode solve_project`, `gate_every 5`,
`batch_pairs 8`, `lr 2e-3`, `epochs 666`, `token_quant_anneal at_knee`, `w_rate 0.05 entropy`,
`byte_ledger_coder smevr`, `telemetry_v9_port on`, the three lane-guard pieces, `basin_handoff off`,
gt cache `gt_n600.npz`, `--full-confirm`. **`ema_decay` is DERIVED by the trainer** from run
geometry (`ema_decay_run_geometry_v1`), never passed — as in the incumbent.

`token_init_mode=solve_project` was verified ds-generic by source read: `tgt.reshape(P, gh, D, gw,
D, 3)` with `384 = 12×32` and `512 = 16×32` both integer. Third ds-dependent surface; no blocker.

### 4.2 Governed launcher — dry-run receipt

`tools/launch_tr1_run.py --dry-run` (G1 seal freshness / G2 import custody / G3 memory / G4 scorer
slot / G5 detached+receipted):

```json
{ "venv_custody_gate0": "PASS",
  "seal_freshness":     "PASS",          // recompiled argv == sealed argv
  "import_custody":     "/Users/adpena/Projects/pact/src/tac/__init__.py",
  "memory_free_gib":    98.9,
  "memory_floor_gib":   25.6,
  "scorer_slot":        "FREE",
  "scorer_slot_near_misses": 0 }
DRY-RUN OK (all gates pass; not launched)
```

### 4.3 Memory preflight — the anchor is a surrogate, so I measured at the true config

**G3's 25.6 GiB floor is `tb1` T2's MEASURED 12.8 GiB × 2.0 — taken at D16.** That is a *different
geometry*. Per the charter and the `#205` lesson (a throughput gate passed at B=8, then the real
config OOM'd at 90 GB entering the n600 loop and died with no checkpoint), I ran the **true ds=32
config** — real n600, real `gt_n600.npz`, every lever on:

```
epoch  0  loss 66.6605  gnorm 11.716  t_wall  43.4  weights_stepped True
epoch  1  loss 18.6421  gnorm  9.882  t_wall  81.3  weights_stepped True
epoch  2  loss 15.1467  gnorm  8.844  t_wall 118.5  weights_stepped True
epoch  3  loss 13.5742  gnorm 18.852  t_wall 155.7  weights_stepped True
epoch  4  loss 12.4692  gnorm 11.992  t_wall 191.7  weights_stepped True
a1_gate   epoch 4  FIRST_GATE  realized_gate_dseg_mean 0.032284065529152195
                   renderer_bytes 4028   total_counted_bytes 49153
canary    epoch 4  PASS: "synthetic decoupling FIRES TRAIN_VERDICT_DECOUPLING, a known-effect
                   d_seg descent is positively registered, and the descending run stays alarm-quiet"
epoch  5  loss 11.7239  gnorm  9.399  t_wall 239.7  weights_stepped True
```

**It trains.** Loss monotone 66.66 → 11.72, `weights_stepped: True` every epoch, a real realized-argmax
gate at ep4, and the confound-immune-system's **positive-control canary green at ds=32** (the
instrument registers a known effect, so the run's own telemetry is trustworthy — CLAUDE.md's L3
verdict-clearance layer).

Two in-vehicle numbers worth banking:

* **`renderer_bytes 4028`** independently confirms gd3 §2.2's MEASURED ds=32 renderer section
  (4,085 B) to within **1.4%** — from a real trained module, not a synthetic one. gd3's
  "+744 B measured, not +778 B derived" correction holds.
* **`total_counted_bytes 49,153`** at ep4 (tokens ≈ 45,125 B). This is a **floor, not a
  prediction**: at ep4 the token field is still near the smooth `solve_project` projection and its
  entropy rises through training. Recorded so the endpoint can be compared against something real.

**Memory — CERTIFIED PEAK, and it REFUTED my own projection.** A 1 Hz sampler run across a full
from-scratch ds=32 run through the ep4 gate:

```
PEAK_RSS_KIB=15743040   PEAK_RSS_GIB=15.014      (true config, n600, all levers, through the a1 gate)
tb1 T2 D16 MEASURED anchor                12.8 GiB
launcher G3 floor (= 12.8 x 2.0)          25.6 GiB   free at dry-run: 98.9 GiB
```

**ds=32 peaks at 1.17× the D16 anchor — HIGHER, not lower.** While the sampler was in flight I had
written that the direction was "unambiguous" because the ds=32 token field is 4× smaller and the
extra conv sits at the smallest resolution. **That reasoning was wrong and the measurement refutes
it**: the 7-conv stack adds an activation and an autograd-tape stage, and that outweighs the smaller
token field. An earlier 5.01 GiB spot-sample (taken mid-epoch-1, before the gate) would have
under-reported the peak by 3×. This is precisely the `#205` genus — a smaller/earlier sample handing
out a false green — and the only reason it did not propagate is that the charter required a real
measurement at the true config.

**Verdict: the launch is SAFE but the launcher's floor is now known to be optimistic for this
geometry.** 15.014 GiB peak against 98.9 GiB free clears with a 6.6× margin, so `window_01` is safe
to fire as sealed. But G3's floor is *derived from the D16 anchor*: the geometry-correct ds=32 floor
at the same 2.0 safety factor is **30.0 GiB, not 25.6 GiB**. That still passes today; it would not
be authority on a loaded host. **OWED (§8): re-anchor `launch_tr1_run.MEASURED_T2_PEAK_RSS_GIB` per
geometry rather than as a single D16 constant** — a D16 constant standing in for every geometry is
the borrowed-constant genus.

### 4.4 Resumability — OBSERVED, not read

CLAUDE.md's P0: resumable-from-disk, per-stage checkpoints under **distinct stage-encoded**
filenames, **EMA shadow** saved, atomic write, loop-end-only saving FORBIDDEN. Verified by watching
the ds=32 run write them:

```
/Volumes/VertigoDataTier/pact/ddm_gd4_ds32_20260803/memprobe/checkpoints/
  stage_solve_init_pretrain.npz        3,928,776 B   <- stage boundary
  intra_seg_trunk_ce_ep00004.npz       3,930,348 B   <- periodic intra-stage
```

Distinct stage-encoded names (no overwrite of the prior stage), intra-stage periodic saves, and
`save_checkpoint` writes `param::` + **`ema::` (the shadow)** + `opt::` + `meta::` via tmp+`os.replace`.
The §1.3 guard now makes the *reload* side fail-closed as well, which is the half that was missing.

---

## §5 G3 — THE POSE RE-SOLVE, BUDGETED INTO THE ROW

gd3 §3 falsified mt1's "ΔS pose = 0": `frame_0 := a·warp(f1) + b` with per-pair `(a,b)`, `s_t`,
plane selector and rolling-shutter `beta` **all solved at encode time against the ds=16 decoded
frame_1**. A ds=32 run emits a different frame_1, so the whole fitted carrier goes stale. The row
costs **1 training run + 1 pose re-solve.**

**The live executor is `tools/ddm_v4d_resolve.py`, mode `refine`** — per-pair dim0 offset-lattice
re-solve → `(a,b)` 2-param GN re-fit at the new dim0 → `beta` select over {0.0, 0.5, 1.0}, each step
**realized-accepted through the real receiver + frozen PoseNet** (accept only Δd ≤ 0), emitting a
**resumable JSONL**, then `tools/ddm_v4d_build_composed_archive.py`. Chain:
`ddm_v4c_resolve.py → ddm_v4d_resolve.py → ddm_v4d_build_composed_archive.py`.

**Correction to the charter's hint (do not inherit it):** `#850`'s truncated-GN relinearization cap
lives in `solve_terminal_pose_gn`, whose only callers are `pb1_*`, `rehearse_*` and its own tests —
**`ddm_pw1` measured that path DEAD on this vehicle** (`manifest.pose_carrier =
two_plane_static_photo_beta_v4d`; `pb1` is unreferenced by any v4d script). So the "don't inherit
the 2–3-relin cap" instruction targets a tool the ds=32 re-solve will not run. The live `refine`
path's analogue is `_refine_dim0`'s bounded grid (`d0 ± 0.048` coarse, `± 0.006` fine) — **that** is
the truncation to interrogate, and `ddm_pw1` separately flags that selector/beta are *not*
re-solved at the refined dim0/(a,b) operating point. Both are live headroom, neither is a blocker.

**Two live inputs that make this cheaper than a cold solve:**

1. **`ddm_pj2`:** `t = s_t·[p2, p1, p0]` with `s_r = 0 ⇒ R = I`, an **exact multiplicative
   degeneracy** (rel homography diff 4.539e-16, n600, two independent derivations). The pose
   translation and `s_t` are **one coordinate**, so the re-solve does not need an independent `s_t`
   search — it folds. (This is also the mechanism `ddm_dc1` used to correct my sister-arm's
   `ms8` story: the win was degeneracy, not dead-codeword reachability.)
2. **`ddm_pb2`'s decode adapter** reconstructs `(a,b)` bit-exact, selector exact and beta exact
   without a full decode — explicitly built so "every future pose re-solve" stops paying that cost.

**Scheduling constraint (binding):** `refine` is REALIZED — it runs the frozen PoseNet and therefore
**takes the single n600 scorer slot**. It must be sequenced *after* the ds=32 training reaches an
endpoint worth composing, and coordinated through MAIN against `ddm_pj2`'s gate. Budget the row as
**training windows → endpoint → `ddm_v4d_resolve --mode refine` (scorer slot) → compose → byte-close
→ exact eval**, not as "train, then see."

**I claim no post-re-solve d_pose magnitude.** gd3's 76.19 is a stale-carrier number and says
nothing about the re-solved value; I have added nothing that would.

---

## §6 G4 — THE FALSIFIER, PRE-REGISTERED BEFORE THE RUN

Written down now, against the arm actually sealed. Not to be softened after seeing the curve.

**PRIMARY.** `grid_downsample=32` pays **iff** the trained ds=32 endpoint's realized
`Δd_seg` above the live `dc1_fold` base `0.00431179` is

> **Δd_seg < 1.633e-03**

for the sealed **`any`** mask (ΔS_rate = **−0.163310**). If the arm actually run is `stride2`, the
threshold is gd3's **1.722397e-03**; if `all`, **1.812e-03**. Quote the threshold belonging to the
mask that ran — **1.633e-03 for the sealed ticket.**

Equivalently: endpoint realized d_seg must land **below 0.00594** (= 0.00431179 + 1.633e-03).
For scale, 1.633e-03 is **37.9%** of the live d_seg; the ep4 first-gate reading was 0.032284, i.e.
**7.5× base**, as expected from a from-scratch run five epochs in — that number is a start-of-curve
datum and is **not** evidence about the endpoint.

**SECONDARY (inherited from gd3 §7, restated for the sealed arm).**
2. If the native ds=32 run emits `archive.zip` **≥ 187,000 B**, the row is under half its claimed
   size and the ordering against `code_width` must be re-derived. **§2.3 already shows the maskless
   arm lands at 189,340 B and trips this** — so this falsifier is *live*, not hypothetical, and the
   mask choice is what keeps the row on the right side of it.
3. If a ds=32 run with a **re-solved** pose carrier lands d_pose within `0.00516576 ± 20%`, mt1's
   "pose ≈ 0" was right in effect though wrong in mechanism, and the re-solve cost gd3 and I both
   added to the row is overstated.

**FALSIFIERS FOR THIS UNIT (mine, not inherited).**
4. If `up4` is later shown to receive gradient but be functionally bypassed (e.g. its output
   dominated by the skip/identity path such that endpoint quality at ds=32 matches a 6-conv ds=32
   ablation), then my §1.2 causal control was necessary but not sufficient and the inert-lever
   question reopens at the *endpoint* rather than at init.
5. If a trained ds=32 field turns out to leave far more than 109 cells effectively constant, the
   `any` mask is over-provisioned, its 22.5% is pessimistic, and the mask should be re-derived from
   the ds=32 run's *own* flip-mass field rather than pooled from the ds=16 artifact.
6. If the ds=32 endpoint's peak RSS drifts materially above 15.014 GiB as the token field's entropy
   rises through training (my certification is a *from-scratch through-ep4* peak, and the gate is the
   heaviest step I observed — but I did not observe a post-knee `tau_softplus` gate), the
   geometry-correct floor in §8.1 is itself too low.
7. If `assert_resume_geometry_compatible` ever refuses a resume that a human confirms is legitimate,
   the guard is too strict and the shape-conflict rule needs an explicit allow-list — I found no
   such case in 672 real checkpoints, but one point is not a curve.

---

## §7 WHAT I DID **NOT** DO (scope honesty)

* **Did not fire the heavy run.** MAIN fires. `--dry-run` only; nothing detached, no `launch_receipt`
  written for `window_01`. The two multi-epoch probes are memory/throughput preflight into
  `memprobe*/` dirs, explicitly named so they cannot be mistaken for the row's run.
* **Did not certify a peak RSS** (§4.3). Sampled, not certified. Named as OWED.
* **Did not touch archive layout or migration** — `ddm_ix2` owns that, including gd3 §4.3's finding
  that `manifest.tokens_sha256` matches neither payload and is never read (`#417` counted-but-inert).
  I neither wrote nor relied on that field.
* **Did not do composition** (`ddm_cp1`) or rasterization (`ddm_ra1`).
* **Did not design the epoch schedule with `ddm_tl1`.** The ticket carries the incumbent's 666-epoch
  / 30-min window verbatim. At the MEASURED **~37–48 s/epoch** at n600 ds=32, 666 epochs is **~7–9
  h**, and a 30-min window advances only ~40 epochs — so the from-scratch ds=32 run needs **many**
  governed windows where the incumbent's window_01 was already a continuation. **`ddm_tl1` should
  size this before the long burn** (scored space ≤10 dims/site: pose `rank(J) ≤ 6` measured, seg
  head affine rank-4) — the row may not need 666 epochs from scratch at all.
* **Ran no scorer-authority evaluation.** Every number here is `[macOS-CPU/MLX advisory]`.

---

## §8 NEXT-IF-RESUMED

1. **Re-anchor the launcher's memory floor per GEOMETRY** (§4.3). The peak is now CERTIFIED at
   **15.014 GiB** for ds=32 vs the D16 anchor's 12.8 GiB, so `launch_tr1_run.MEASURED_T2_PEAK_RSS_GIB`
   under-projects this geometry by 17% and its 25.6 GiB floor should be 30.0 GiB here. Safe today
   (98.9 GiB free) — not safe as a standing constant. One constant standing in for every geometry is
   the borrowed-constant genus.
2. **Have `ddm_tl1` size the epoch budget** before MAIN fires the long burn (§7). The sealed window
   is the incumbent's, not a derived one.
3. **Fire `window_01`** via `tools/launch_tr1_run.py --ticket …/ticket_ds32_window_01.json --out-dir
   …/ddm_gd4_ds32_20260803/window_01` (all gates already PASS), then chain windows with
   `--resume-from` — now fail-closed against every one of the 672 ds=16 checkpoints.
4. **At the endpoint: pose re-solve** via `ddm_v4d_resolve --mode refine` (§5) — sequence it against
   the single n600 scorer slot through MAIN. Interrogate `_refine_dim0`'s `±0.048/±0.006` bounds and
   `ddm_pw1`'s "selector/beta not re-solved at the refined operating point"; do **not** chase
   `#850`'s cap, which is dead code here.
5. **Adjudicate the falsifier at §6's 1.633e-03** — the threshold for the mask that ran.
6. **OWED, not done:** a STRICT preflight catalog gate for the resume-geometry class. The runtime
   guard is fail-closed and tested (10 tests incl. 4 parametrized geometry cases and a
   "still allows the legitimate resume" case), which is the self-protection that binds at the
   surface where the bug lives; the static sister gate is the second landing CLAUDE.md's
   two-landing rule asks for and I did not claim a catalog number for it.

---

## §9 CUSTODY

| artifact | detail |
|---|---|
| `…/ddm_gd4_20260803/ticket_ds32_window_01.json` | **hash `d5ba6b30f1df152e4e8478fa2cb575970bb2c557588f16e4c32ef943a5020b2b`** |
| `…/ddm_gd4_20260803/ticket_ds32_memprobe.json` | hash `b699d44cdfafe69d86608a1c28dba513476cc1dae9d832a01f7fa3eac219e09e` |
| `…/ddm_gd4_20260803/qa24_grid_keep_mask_50_D32_{any,stride2,all}.npy` | 109 / 94 / 78 of 192 kept |
| `…/ddm_gd4_20260803/ds32_cell_mask_provenance.json` | source sha256 + rule + the four measured arms |
| `…/ddm_gd4_ds32_20260803/memprobe/{telemetry.jsonl,tr1_config.json,checkpoints/}` | 19 rows, ep0–5 + a1 gate + canary; 2 per-stage checkpoints |
| `…/ddm_gd4_ds32_20260803/memprobe3/` + `…/ddm_gd4_20260803/peak_rss2.log` | 1 Hz peak sampler through the ep4 gate: **PEAK_RSS_GIB=15.014** |
| code | `experiments/train_tr1_partition_renderer_mlx.py`, `src/tac/witness_dsl/spec_tr1_renderer_20260728.py`, `src/tac/tests/test_ddm_tb1_tr1_renderer.py` (+10 tests; 41 pass) |

**No candidate gated. No pointer moved. `score_claim=false` on every row above.**
