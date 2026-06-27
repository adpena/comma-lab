# Yousfi levers — RECURSIVE ADVERSARIAL REVIEW + OPTIMAL-FORM pass (level-set witness)

**UTC:** 2026-06-27T06:33:35Z · **Author:** levers optimal-form review subagent (DAG **FEED-df**) ·
**Axis:** `[macOS-CPU advisory] / [macOS-MLX training-gradient]` NON-PROMOTABLE · **Pointer:** UNMOVED 0.19110.
**Constraints honored:** CPU-only `$0` (GPU pid 72600 mod-32 baseline UNTOUCHED, confirmed alive 2:58+ under
safe_run.py), additive/default-off edits only, serializer + `--expected-content-sha256` commits, review-gate
mark-file, subagent_checkpoint, no `/tmp` in artifacts.

Reviews the 3 Yousfi levers landed 2026-06-27 (commit `3ba748a18`, FEED-dc) per CLAUDE.md
"Recursive adversarial review protocol" (3 consecutive clean passes) + "Substrate MUST be at OPTIMAL FORM
before paid empirical dispatch". Files: `experiments/train_levelset_witness_realized_through_R_mlx.py`,
`src/tac/boundary_math/lever_b_levelset_generator.py`,
`src/tac/boundary_math/tests/test_levelset_yousfi_levers.py`.

---

## PART A — 3-CLEAN-PASS RECURSIVE ADVERSARIAL REVIEW LOG

### VERDICT: **SEALED** (3 consecutive clean passes A/B/C after R1+R2+R3 fixes). Counter: 3/3.

Fixes landed: `7bb37c18a` (R1) + `68d71f2f8` (R2) + `853de4f7a` (R3 class-order self-correction). Tests
11/11 lever + 19/19 sister generator = 30/30 pass (MLX-free); preflight ZERO new failure on touched files
(the `CodebaseDriftError` is the pre-existing unrelated `experiments/launch_*.py` drift; my files NOT
implicated). GPU pid 72600 untouched throughout (3:19+).

| Round | Adversarial perspective | Findings | Result |
|---|---|---|---|
| R1 | callsite-trace + class-order(NO-FAKE) + stage-interaction + aliasing-claim + byte-close gap | **3** | NOT clean → fixed `7bb37c18a` |
| R2 | edge-case / degenerate-config mental execution | **1** | NOT clean → fixed `68d71f2f8` |
| R3 | deeper class-order cross-check (DAG FEED-da luma-sort vs frozen_source) | **1** (my own R1 "fix" had asserted the LESS-supported order) | NOT clean → fixed `853de4f7a` |
| A | re-read all edited regions: coherence + comment-vs-code parity | 0 | **CLEAN #1** |
| B | NO-FAKE re-audit (does each lever do its named work on REAL inputs?) | 0 | **CLEAN #2** |
| C | assumption-challenge + determinism + full suite + preflight | 0 | **CLEAN #3 → SEAL** |

The R3 finding is the recursive protocol working as designed: questioning my own R1 interpretation surfaced
that I had "corrected" the comment to the wrong side of a genuine inter-memo dispute. Net resolution below.

### CLASS-ORDER RESOLUTION (the R1→R3 arc — the load-bearing fact for the lever)
- **CONFIRMED beyond dispute: class 0 = Road, class 1 = Lane** (all project memos agree). **The lever uses
  ONLY class 1** (`--lane-edge-class 1` default) → the lever's lane targeting is CORRECT and was correct in
  every commit. This is the only class fact the lever depends on.
- **Classes 2/3/4 were genuinely DISPUTED across project memos** and I mishandled it: R1 "corrected" the
  comment to `[Undrivable2,Movable3,MyCar4]` (the `frozen_source_0byte` static-mix labels). R3's deeper
  cross-check found the **code-grounded PIL-luma sort** is more authoritative: comma10k mask colors →
  PIL-L luma = {Road 42, Lane 76, MyCar 90, Undrivable 124, Movable 161}; the SegNet indexes by sorted
  `class_values [42,76,90,124,161]` → **`[Road0, Lane1, MyCar2, Undrivable3, Movable4]`** (matches DAG FEED-da
  + `order_exploit` + the ORIGINAL FEED-dc comment). **R3 fix (`853de4f7a`)** reverted to the luma-correct
  order and honestly flags 2/3/4 as disputed-not-load-bearing (DO NOT use `--lane-edge-class 2/3/4` without
  resolving against the actual SegNet `class_values`, which is NOT vendored in-repo).

### Round 1 findings (FIXED, `7bb37c18a`; finding 1 superseded by R3 `853de4f7a`)
1. **[Medium→R3] Class-order comment.** R1 flagged the 2/3/4 comment labels as a future-config trap and
   "corrected" them — but to the wrong side of the dispute (see CLASS-ORDER RESOLUTION above; R3 fixed it).
   Lane=1 (the lever's only dependency) was always correct.
2. **[Medium] Missing `--lane-edge-start-epoch` (stage-interaction / optimal-form).** The lane margin hinge
   engaged from ep0 over ALL stages incl. the coarse `ce` stage → risks the known
   margin-from-scratch-starves-interior failure (partition not formed yet). The optimal-form sweep
   ("ep0 vs gated-to-l7") was NOT EXECUTABLE. **Fix:** added `--lane-edge-start-epoch` (default 0 = current
   behavior; gate to e.g. 300 to align with the tau_softplus margin-sharpening stage).
3. **[Medium] Engagement transition spike-skip (the named "margin-engage spike-skip" failure).** When
   lane-edge engages mid-training, the added term raises the loss scale → the spike-guard
   (`batch_loss > spike_factor·median`) would silently SKIP the first post-engage batches → the lever never
   applies. **Fix:** the engage epoch RE-TREATS the spike-guard (`recent_losses.clear()` + `lane_edge_engage`
   print) per operator 2026-06-26 "transitions must re-treat". Validated: `$0` CPU smoke (n6/3ep,
   start-epoch 2) → `lane_edge_engage` fired at ep2, ep3 verdict finite, EXIT=0, NO spike-skip.

### Round 2 finding (FIXED, `68d71f2f8`)
4. **[Medium] Silent-no-op config (NO-FAKE silent-skip) + mid-train IndexError.** `--lane-edge-weight>0`
   with `--lane-edge-start-epoch > epochs` → the hinge never engages → a FALSE "lane-edge doesn't help"
   verdict; `--lane-edge-class` outside [0,4] → IndexError mid-training (after GPU spend). **Fix:** extracted
   `validate_lane_edge_config(...)` pure fail-closed helper (raises BEFORE any GPU spend; NO-OP when lever
   off → additive default preserved) + 4 unit tests. Validated: start>epochs → ValueError; class∈{7,-1} →
   ValueError; weight 0 + bad args → no raise (default-off untouched).

### Round 3 finding (FIXED, `853de4f7a`)
5. **[Medium] Self-correction: R1 asserted a disputed class order.** Caught while writing the DAG feed by
   cross-checking FEED-da's code-grounded luma-sort against `frozen_source`. Reverted to the luma-correct
   order; class 1=Lane unaffected. See CLASS-ORDER RESOLUTION above.

### Clean passes A / B / C (after R3 fix)
- **A (coherence + comment-vs-code):** re-read every edited region — the gate helper (return-early-when-off,
  raises on both misconfigs), the lane_gate wiring (capture@399, init@409, branch@430, announce@527, epoch
  re-treat@545-551, all in scope), and the corrected class-order comments. ZERO findings.
- **B (NO-FAKE re-audit):** all three levers do their named work on REAL inputs, not markers.
  CHROMA — `levelset_rgb_forward_numpy` genuinely realizes 3-channel RGB (max|Δ|=209.7 vs achromatic, phi
  invariant). STEM-NYQUIST — `curvelet_directional_B(max_freq)` genuinely drops atoms (test: shape shrinks,
  max-norm ≤ cap); flag help is accurate ("drop curvelet atoms"). LANE-EDGE — the term renders f1→R→
  `adapter.segnet`→live margin→`relu(target-margin)·lane_mask`, a real realized-through-R training gradient
  (CPU-torch verdict is authority; MLX is the legal train-gradient device). No false claims. Comments match
  code (re-checked the new flag/guard comments).
- **C (assumption-challenge + determinism + integration):** the lane-edge lever inherits the assumption
  "up-weighting class-1 lane margin lowers TOTAL d_seg." This is **CARGO-CULTED-adjacent**: CLAUDE.md's
  measured binding residual is the UNION of all-class edges (50% class-0, 19% class-1, 13% class-2); lane is
  only ONE component. Up-weighting lane could trade class-0 accuracy (the 50% majority) for class-1. This is
  a DESIGN risk, NOT a code bug — the code is correct AND the verdict (`cpu_verdict_d_seg_batch`) already
  measures TOTAL d_seg → Falsification criterion (Part B): decision metric is TOTAL d_seg; lane-edge KILLED
  if total d_seg RISES even if lane IoU improves. Determinism: seeds set; lane term + guard add no
  randomness; 30/30 tests pass; preflight no new failure (drift pre-existing, unrelated). `__cfg_lane_edge_*`
  + `__cfg_max_bank_freq` persisted to npz for provenance; `start_epoch` correctly NOT persisted (training
  schedule, not a deploy/inflate parameter). No code finding → clean.

### Pre-existing observations (NOT lever regressions, logged for hygiene)
- Benign `RuntimeWarning: divide/overflow/invalid in matmul` at `lever_b_levelset_generator.py:179`
  (`curvelet_feats`, UNTOUCHED front-end code; output finite; pre-existing in the FEED-dc path). Not in scope.
- Pre-existing `CodebaseDriftError` (11 `experiments/launch_*.py` ad-hoc launchers) — unrelated, not mine.

---

## PART B — OPTIMAL-FORM OPTIMIZATION DESIGN (per-lever own optimum; GPU sweep DESIGNED, not run)

`$0` CPU pre-screen (`--mlx-device cpu`, n6, 3ep) validates MECHANISM only (3ep = noise, NOT efficacy);
efficacy needs the GPU n600 run after pid-72600 frees. **Decision metric = TOTAL realized d_seg (CPU-torch
verdict), never lane IoU alone.**

### ALIASING-CLAIM VERIFICATION ($0 numpy, MEASURED — claim CONFIRMED, stronger than memo)
Self-orient dir feat across-edge channel = `sin(2π·fc·u_n)`, fc = freq_across·2^k. At freq_across=32,
n_dir_freqs=6 → fc ∈ {32,64,128,256,512,1024}. FFT of the sampled feature on the 512-px render grid
(Nyquist 127.75 cyc/unit; SegNet stem Nyquist 64):

| fc (cyc/unit) | observed FFT peak | aliased? | folds to | over stem-Nyq(64)? |
|---|---|---|---|---|
| 32 | 31.94 | no | — | no |
| 64 | 63.88 | no | — | no |
| 128 | 127.75 (= grid Nyq) | borderline | — | **yes** |
| 256 | **0.50** | **YES** | near-DC garbage | yes |
| 512 | **1.00** | **YES** | near-DC garbage | yes |
| 1024 | **2.00** | **YES** | near-DC garbage | yes |

→ fc∈{256,512,1024} alias at the RENDER GRID ITSELF (before R), folding to near-DC noise; fc=128 sits at
grid-Nyquist + above stem-Nyquist. Only {32,64} are clean. **The anti-alias lever is REAL** — but the actual
control is `--n-dir-freqs` (NOT `--max-bank-freq`, which is a curvelet-bank no-op at default; the curvelet
bank already maxes at 16 cyc/unit). `--n-dir-freqs 2 @ freq_across 32` keeps only {32,64}.

### Per-lever sweep matrix

| Lever | Knob | Sweep | Predicted d_seg direction | Falsification threshold |
|---|---|---|---|---|
| **LANE-EDGE** | `--lane-edge-weight` | {15, 30, 50} | mid (~30) optimal; >50 starves class-0 majority → total d_seg ↑ | KILL if TOTAL d_seg ≥ no-lane baseline |
| **LANE-EDGE** | `--lane-edge-start-epoch` | {0, 300, 900} | **300 (tau stage) optimal**; 0 risks coarse-stage starvation; 900 too late | revert to 0 if 300 ≥ 0's d_seg |
| **LANE-EDGE** | `--lane-margin-target` | {0.3, 0.5, 1.0} (secondary) | 0.5 base; higher = more aggressive widen | n/a (2nd-order) |
| **ANTI-ALIAS** | `--n-dir-freqs` (self-orient ON) | {2@freq32, 4@freq8} vs 6 (control) | clean {32,64} ≥ aliased-6 (removes garbage feats) + −0.4–0.7 KB | keep 6 if cleaning RAISES d_seg |
| **ANTI-ALIAS** | `--max-bank-freq` | 64 (defensive no-op) | n/a (curvelet already sub-Nyquist) | — |
| **CHROMA** | `--chroma` | {on, off} | ON ↓ d_seg (SegNet reads chroma-saturated boundaries); FREE bytes | drop if on ≥ off |
| **CAPACITY/RATE** (separate arm) | `--hidden-dim` | {48, 64, 96} | 96 safest for d_seg; 48/64 test capacity cliff (rate NOT binding ~4:1) | the bc20/bc36 trilemma test |

**Interactions:** lane-edge × chroma = likely COMPOUNDING (chroma makes lanes separable, lane-edge sharpens).
anti-alias × self-orient = COUPLED (anti-alias only matters when `--self-orient` ON). lane-edge × anti-alias =
independent (loss vs feature). lane-edge × curriculum = engage at 300 aligns with tau_softplus (both
margin-sharpening) → coherent.

### OPTIMAL n600 LAUNCH CONFIG (deploy these levers; sweep the rest separately)
Base = the running good config (n96→n600, hosc + siren + palette-anchor + curriculum ce→tau300→l7900,
render 384). Levers at their reviewed optima:
```bash
env TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python -u \
  experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir experiments/results/levelset_n600_yousfi_optform_$(date -u +%Y%m%dT%H%M%SZ) \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 --epochs 1500 --render-h 384 --render-w 512 \
  --hidden-dim 96 --mod-dim 32 --activation hosc --siren-init \
  --softmax-temp-start 1.0 --softmax-temp-end 0.05 \
  --curriculum --tau-softplus-start-epoch 300 --l7-start-epoch 900 \
  --palette-anchor --self-orient --reorient-every 50 \
  --freq-across 32 --n-dir-freqs 2 --freq-along 4 --max-bank-freq 64 \
  --chroma \
  --lane-edge-weight 30 --lane-edge-class 1 --lane-margin-target 0.5 --lane-edge-start-epoch 300 \
  --w-seg 100 --w-pose 0 --eikonal-weight 0.01 --length-weight 0.001 \
  --ema-decay 0.997 --accum-pairs 8 --grad-clip 1.0 --verdict-pairs 96 \
  --mlx-device gpu --eval-every 25
```
Wrap with `tools/safe_run.py --rss-mb <cap> --timeout <s>` per the scale-safeguard discipline. **Do NOT launch
while pid 72600 holds the GPU.** Sweep arms (separate dispatches): lane-weight {15,50} × start {0};
n-dir-freqs {4@freq8, 6}; hidden-dim {48,64} (capacity/rate cliff test).

**Optimal-form delta vs FEED-dc's launch:** lane-edge gated to ep300 (was ep0; avoids coarse-stage
starvation + the spike-skip is re-treated) — the single load-bearing optimization-design change.

---

## CHROMA BYTE-CLOSE GAP VERDICT — **the level-set witness has NO exact-eval path = ROW-BLOCKER**

- **Chroma itself byte-closes** (within the trainer's own primitives): `levelset_rgb_forward_numpy` int8-dequant
  round-trip is finite + chroma-aware; `quantize_levelset_blob` measures real bytes and is chroma-INDEPENDENT
  (`out_tex` 3-ch either way) → chroma is a FREE-rate d_seg lever. Chroma is NOT the blocker.
- **The BLOCKER is the missing level-set inflate/byte-close+eval builder.** `tools/witness_byte_close_and_eval.py`
  is hard-keyed to the RGB witness (`params["out.weight"]` at line 143; forward `z = h @ p["out.weight"].T`
  at line 324). The level-set checkpoint (`levelset_witness_ema_mlx.npz`) has `out_sdf`/`out_tex`/`palette`/
  `code` and **no `out.weight`** → the RGB tool KeyErrors. A level-set-specific inflate builder does NOT
  exist anywhere (grep: zero hits). So the levers can produce TRAINED advisory `implied_S` rows
  (trainer-internal `quantize_levelset_blob` + numpy-fp32 CPU verdict) but **CANNOT produce an EXACT-EVAL row
  — the only thing that moves the pointer per THE GOAL.**
- **FIX PATH (separate landing, the imminent-exact-row infra):** build
  `tools/levelset_byte_close_and_eval.py` — mirror the RGB tool but key on the level-set npz
  (`out_sdf`/`out_tex`/`palette`/`code`), regenerate the curvelet bank from `__bank_*` cfg (free table,
  rule 118), thread `__cfg_chroma`/`__cfg_max_bank_freq`, run `levelset_rgb_forward_numpy` (THE ONE CODEPATH)
  over 600 pairs → masks + Quantizr stored-pose sidecar → archive.zip + ≤100-LOC inflate.py →
  `upstream/evaluate.py` (contest-CPU/CUDA). Until it lands, the level-set witness is advisory-only.

---

## Validation summary (NO-FAKE / compliance)
- All edits ADDITIVE / default-off → the running pid-72600 process + any in-flight config unaffected (default
  lane-edge-weight 0 / start-epoch 0 / max-bank-freq None reproduce prior behavior; guard NO-OPs when off).
- `py_compile` OK; 30/30 tests (11 lever incl. 4 new guard tests + 19 sister generator); preflight no new
  failure on touched files; `$0` CPU smokes only (`--mlx-device cpu`, n6/3ep, gt_n6); GPU NEVER touched
  (pid 72600 confirmed alive throughout); numpy-fp32 verdict; smoke scratch cleaned.
- NO score / frontier / promotion / kill claim. Pointer UNMOVED 0.19110. Axis `[macOS-CPU advisory]`
  NON-PROMOTABLE. Commits via serializer (`7bb37c18a`, `68d71f2f8`) with `--expected-content-sha256`.
