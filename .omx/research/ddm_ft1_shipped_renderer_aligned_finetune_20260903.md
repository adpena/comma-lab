# ddm_ft1 — fine-tuning the SHIPPED renderer: the identity gate PASSES on a different object than the charter names, and the charter's training target is the wrong GT lineage

Arm: `ddm_ft1_shipped_renderer_aligned_finetune`. Tokens: `[no-triality] [p0-ledger-ok]`.
Craft contract: `docs/operating_manual_craft_handoff.md`.
Axis of every measured row below: **`[macOS-CPU advisory]`** unless it cites the T4 receipt. No score claim.

## ANSWER FIRST

1. **The identity gate PASSES — on an object the charter mis-identifies.** The frontier archive's
   `semantic_renderer` section is **36,130 B, sha `17e0fd0b…`, SM3R v1 `MODE_ROW_PRUNE_MIXED`**
   (width 96, `keep_percent` 1, per-tensor depths {3,4}), not the charter's "30,856 B, sha
   `39d1be52…`" uniform-int4 section. That section belongs to the superseded **gb1-generation RX1M**
   container (`experiments/ddm_ni1_nr1_k32_receiver_distortion.py:255` still pins its header
   `(b"RX1M", 1, 2, 0, 26, 13_515, 30_856, 22_010)`). Re-derived from the primary artifact, the gate
   passes on the real object: decode → deployed SM3R re-encode is **byte-identical**, state round-trip
   `max_abs_delta 0.0`, and the shipped receiver's forward matches the lifted trainer's forward **and**
   the exact-R path **bit-identically** on 8 spread pairs.
2. **The charter's training target is the wrong GT lineage, and this would have silently mis-aimed the
   whole arm.** `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` is the **PyAV** lineage, not DALI.
   Measured: its `lstars` differ from `gt_cache_av.pt` at **2** of 117,964,800 positions and its
   `gt_poses` by MSE **3.6e-12**, while DALI-vs-PyAV differ by **20,671** argmax positions and pose MSE
   **1.4061e-04** — exactly the additive fork `rf1` records. Shipped tokens vs **DALI** = **9,179**
   mismatches (mst1's DALI-gated reference: 9,182); vs PyAV = **28,133**. Training against the PyAV
   table would have aimed the renderer at **20,671** positions the contest does not score — **87% of
   d_seg's entire budget of 23,757 flips**, and **2.25x the shipped token error itself**.
3. **The charter's promote gate is internally inconsistent by 5.4×.** Its pose guard `d_pose ≤ 1.25e-4`
   costs `√(10·1.25e-4) − √(10·6.37e-6)` = **+0.02737 S**, which is **5.44× the entire credit of the
   25% seg cut it predicts** (0.005035 S). At ΔB = 0 the binding pose ceiling is **≈1.69e-5** (2.66×
   base), not 1.25e-4.
4. **The charter's falsifier threshold sits above the measured family ceiling.** `msr1`'s flow-balance
   ceiling for boundary-moving actuators — a family it states explicitly includes "learned renderer
   weights" — is **2,123 net pixels = 8.94% of d_seg**. The charter fires its falsifier below a **10%**
   fall. On prior evidence the falsifier is more likely than not to fire, and I record that **before**
   the result, not after.
5. **The run is correctly aimed, and its first epoch is a NEGATIVE.** Step-0 n600 advisory baseline on
   the DALI table is **0.00020386589898003471** against the contest-CUDA T4 receipt's **0.00020139** —
   **1.23% agreement**, independent confirmation that the DALI target, the exact-R path and the
   identity gate are all right. At **step 600 d_seg ROSE to 0.00026752895779079860, +31.23%**
   (+7,510 flips, 12.4× the measured 605-flip A/A noise floor; +0.006366 S on the seg leg). **The
   charter's falsifier fired at the first evaluated epoch**, in the worse direction than it
   anticipated, exactly as §8 pre-registered. Scope is **INSTANCE** — this init, lr 2e-5, 1,800-step
   cosine, seg-only loss — **not** a family verdict, because the pose term was never built and the LR
   is the leading suspect (§5.1).

6. **THE ARM'S RESULT: the renderer→(seg,pose) coupling is MEASURED at 217.30**, and it closes the
   seg-only formulation by arithmetic. `|Δd_pose|/|Δd_seg| = 217.30` on an n200 seeded-random draw
   through the shipped receiver — against `rf1`'s independently measured **166.8** for an un-retrained
   structural change. A 25% seg cut therefore drags d_pose to **1,718× base**; even the best measured
   n600 carrier recovery (8.0×, jg5) leaves it **81× over** the 1.694e-05 ceiling that the seg win can
   pay for, and a renderer change has **no per-pair admission lever**. Measured candidate ΔS =
   **+0.3857**. Scope: FORMULATION (seg-only loss, no pose term). The **joint** formulation stays open
   — 217.30 is the coupling of the seg-only gradient *direction*, and a pose-priced loss searches for a
   different one.

**Pointer: UNMOVED.** No exact row was bought by this arm.

## 1. IDENTITY GATE RECEIPT

Receipt: `/Volumes/VertigoDataTier/pact/ddm_ft1_shipped_renderer_aligned_finetune/retained/identity_gate_receipt.json`
Producer: `experiments/ddm_ft1_identity_gate_and_caches.py` (commit `77af37116`).

| leg | measurement | result |
|---|---|---|
| archive | `…/ddm_g8s_single_run_reproof/store_v2/retained/archive.zip` | 180,002 B, sha `cbb8d928…` = `effective_frontier.archive_sha256` |
| section | semantic_renderer, read by the shipped receiver's own parser | **36,130 B**, sha `17e0fd0b197ac147afe98397ef38f02f7915b69372d03c042e6be6fa0f992e50` |
| format | `SM3R` v1 `MODE_ROW_PRUNE_MIXED`, `keep_percent` 1 | width 96, blocks 4, frame_dim 8, 66,339 params, 15,363 zero (23.16%) |
| prune | `blocks.{1,2,3}.film.weight` | **2 of 192 rows kept** each |
| depths | `blocks.0.film.weight` 3, `frame_embed.weight` 3, other 14 tensors 4 | recovered from the payload, not assumed |
| **weights leg** | decode → `pack_prune_mixed_candidate(keep=1, depths=recovered)` | **BYTE-IDENTICAL**, state round-trip `max_abs_delta 0.0` |
| **render leg** | shipped `SemanticTokenRenderer(96)` vs lifted trainer renderer, 8 pairs (0,1,2,137,299,300,450,599) | raw forward **bit-identical** (Δ 0.0); exact-R **bit-identical** (Δ 0.0) |

**The size is value-independent.** `pack_prune_mixed_candidate` lays the payload out from shapes,
`keep_percent` and the depth table — never from the weight values. A fine-tuned state therefore exports
at exactly 36,130 B by construction. Regression-tested
(`test_sm3r_export_size_is_independent_of_the_weight_values`).

**Why the charter's numbers are stale, stated plainly.** 30,856 B / `39d1be52…` is a real object — it
is the renderer in the **gb1** generation's RX1M container. The AFR1 archive that holds the frontier
carries a later, larger, mixed-precision row-pruned representation. This is memory
[[binding-instruction-numbers-expire-and-nobody-rederives-them]] firing exactly as written: the numbers
were correct when they were written into an instruction and nobody re-derived them at the pointer move.

## 2. THE GT-LINEAGE CORRECTION (the load-bearing finding)

The charter directs training against `gt_n600.npz` and calls it "the T4-scored lineage — #1142". Traced
to source: `tools/build_shared_gt_cache_for_mlx_fleet.py` → `precompute_gt` →
`src/tac/boundary_math/seg_core.py:80 decode_gt_frame1_pairs`, which is `av.open(...)` +
upstream `yuv420_to_rgb`. **PyAV container decode.** It avoids the rgb24 trap, but it is not DALI.

Measured against both retained tables (`/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/`,
built on a **Tesla T4 with CUDA** per its own `result_summary.json`):

| comparison | positions differing | rate |
|---|---:|---:|
| `gt_cache_av['seg']` vs `gt_n600['lstars']` | **2** | 1.70e-08 |
| shipped-weight base d_seg, n24 **seeded random** draw | 2.0239e-04 | **0.50%** from the T4 receipt |
| shipped-weight base d_seg, n8 **prefix** draw (superseded) | 1.7230e-04 | 14.5% off — the prefix trap, measured |
| `gt_cache_dali['seg']` vs `gt_cache_av['seg']` | **20,671** | 1.752e-04 |
| `gt_cache_dali['seg']` vs `gt_n600['lstars']` | **20,671** | 1.752e-04 |
| shipped tokens vs **DALI** | **9,179** | 7.781e-05 |
| shipped tokens vs PyAV | 28,133 | 2.385e-04 |
| pose MSE: `gt_cache_av` vs `gt_n600['gt_poses']` | — | **3.57e-12** |
| pose MSE: DALI vs PyAV | — | **1.4061e-04** (= rf1's stated additive fork, to 5 s.f.) |

**Independent corroboration inside the repo.** `experiments/ddm_up2_shipping_pose_solve.py:76-79` —
the terminal pose solver this arm's own fire order calls — already draws exactly this distinction:
`DEFAULT_DALI_GT = /Volumes/VertigoDataTier/.../gt_cache_dali.pt` and
`DEFAULT_AV_GT = experiments/results/mlx_fleet_gt_cache/gt_n600.npz`. **The shipping pose solver calls
gt_n600.npz the AV cache.** The repo already knew; only the charter's label was wrong.

`mst1`'s DALI-gated transmitted-error count for this object is **9,182**; I measure **9,179** on the
AFR1 tokens. Three pixels apart — the token field is essentially unchanged from dx2 to afr1, and the
28,133 figure was never token error at all; it is dominated by GT-lineage disagreement.

**Self-correction, recorded rather than quietly fixed.** An earlier draft of this memo and two code
comments carried **18,954** for the DALI-vs-PyAV fork. That number was never measured: I obtained it as
`28,133 - 9,179`, the difference of two mismatch COUNTS, which is not the size of the symmetric
difference between two label tables -- the two disagreement sets overlap. The measured value is
**20,671**. Corrected in `experiments/ddm_ft1_identity_gate_and_caches.py`,
`experiments/ddm_ft1_verdict_bhw_pose.py` and here; the superseded figure survives only in commit
messages `77af37116` and `8865d5e65`, which are append-only history. This is the operating manual's
Sec 4 failure (recognizing a plausible number instead of re-deriving it) caught by its own Sec 6
(attack your own conclusion).

**Consequences carried forward:**
- The target cache is now `gt_cache_dali.pt['seg']` (sha `a91d9825…`); the PyAV table is retained as
  the labelled control.
- The instrument's own numbers move into the contest regime when the lineage is fixed. Same code,
  8 pairs: on PyAV `d_seg 2.9055e-4 / d_pose 1.956e-4`; on DALI `d_seg 1.7230e-4 / d_pose 3.2115e-6`.
  The T4 receipt is `2.0139e-4 / 6.37e-6`. **d_pose goes from 30.7× off to 0.50× off.** That is a
  validation of the carrier composition, not a coincidence.
- **`∂S/∂d_pose` does NOT transfer across the fork** (626.5 at DALI vs 130.4 at PyAV, 4.80×). Any pose
  weight calibrated on a PyAV advisory row under-weights pose by 4.8×.

**A new measurement this unlocks:** shipped token error (DALI) 7.781e-5 vs shipped d_seg 2.0139e-4 —
**the render AMPLIFIES the transmitted error 2.588×.** Consistent with mst1's 90.4702% manufactured
share, and it is the mechanism warrant for a renderer fine-tune at all.

## 3. WALL-CLOCK AND THE DERIVED BUDGET

No CPU s/pair existed for any renderer trainer, so I measured three points and fitted the CE1-form
cost model (CPU, 4 threads, M5 Max, with the governed QBR1 Metal burn co-resident).

| run | steps | pairs | wall |
|---|---:|---:|---:|
| A | 6 | 8 | 52.15 s |
| B | 26 | 8 | 124.78 s |
| C | 6 | 64 | 180.20 s |

`total_s = F + r·steps + e·(pair-evals)` with **F = 12.068 s**, **r = 3.6315 s/step** (batch 1),
**e = 1.1433 s per pair-eval**. Three points, three parameters — **exactly determined, no out-of-sample
validation**; labelled DERIVED, not MEASURED-with-residual. (CE1's MPS fit was `r = 0.22267 s/step`;
CPU is **16.3×** slower per step.)

Derived budget:

- 1 epoch = 600 steps (batch 1, w96b's shape) = **2,178.9 s = 36.3 min**
- 1 n600 advisory eval = **686.0 s = 11.4 min**
- w96b's 65-epoch reference window = **51.7 h** — not deliverable inside this arm.
- **Chosen window: 1,800 steps (3 epochs), eval + checkpoint every 600 → 9,293 s ≈ 2.58 h.**

## 4. CONFIG AND ITS DERIVATION (every constant, with its warrant)

| knob | value | derivation |
|---|---|---|
| loss | `sigmoid(−(z_target − max z_other)/τ).mean()` on R→uint8→SegNet logits | `semantic_renderer_oracle.py:181-194` (sha `ffdf0988…`), the sealed CE1/w96b law. The 100× in w96b's writing is inert under AdamW (rg1b) and absent from this trainer. |
| curriculum | `--ce-fraction 0.0 --softplus-fraction 0.0` | w96a's sealed row: 100% `expected_flip`, the measured best-aligned of CE1's three objectives. |
| τ | linear **0.15 → 0.05** across the whole window | same source law; with `softplus_fraction 0.0` the tail spans 100% of the run. **Declared risk:** τ is a fraction of the run, so a 1,800-step window anneals τ **21.7×** faster per step than w96b's 39,000. CW1 names this confound. |
| optimizer | AdamW, weight decay 0 | trainer line 1065. |
| LR | **2e-5**, cosine to `0.01·lr` | CE1's measured aligned plateau (6e-5 weaker, 1e-5 null). The 1% floor is already this trainer's (`:1066`). The alternative rung, 2e-7, is this checkpoint's own PR130 stage-08 tail LR (BS16: −3.03% in 30 steps at batch 16) but is the **token**-objective tail and rc2 measured it "did not train" in a comparable window. Integrated LR = `steps·0.505·lr` = **0.01818**, 60% of EF3000's 0.0303, the smallest budget that has ever produced a real aligned descent. |
| batch | 1 | w96b's shape; 600 steps = 1 epoch over n600. |
| target | `gt_cache_dali.pt['seg']` | §2. |
| input | shipped decoded `tokens.u8` (sha `cc10a7b0…`) | the renderer's actual conditioning. |
| class weighting | **none** | bz2d measured the shipped renderer's flips as a *uniform field-quality deficit* across four of five classes inside a 1.31× band: "there is no class to fix and recover." The margin loss already weights by proximity-to-flip, which is a site weighting. |
| `--fixed-zero-mask` | **off** | it masks every zero-valued weight (23.16% of the model), not just the pruned rows. Freezing a quarter of the model is a capacity cut the export does not require. The cost of leaving it off is the realization gap in §6, which is measured rather than assumed. |

## 5. PER-EPOCH TABLE

Run: `/Volumes/VertigoDataTier/pact/ddm_ft1_shipped_renderer_aligned_finetune/runs/aligned_dali_lr2e5_s1800/`
(detached, `--nice 10`, per-eval EMA checkpoints, `--resume-from` available).
`quantized_exact_seg` = the trainer's own n600 advisory d_seg through the exact-R path against the
DALI table, on the EMA shadow. **Advisory, `[macOS-CPU advisory]`, never a score.**

| step | epoch | advisory d_seg (DALI, n600) | flips | vs step 0 |
|---:|---:|---|---:|---:|
| 0 | 0 | 0.00020386589898003471 | 24,049 | — |
| 600 | 1 | 0.00026752895779079860 | 31,559 | **+31.23%** |
| 1200 | 2 | 0.00023271348741319445 | 27,452 | **+14.15%** |

`best_quantized_exact_seg` is still **step 0's** value at both rows: not one evaluated point improved
on the shipped renderer.

**The excursion is recovering, and it tracks the LR anneal.** The expected-flip loss falls
0.0012238547 → 0.0003395645 (−72.3%) while lr anneals 1.505e-05 → 5.150e-06 (−65.8%), and d_seg gives
back **54.69% of its excursion** (peak +7,510 flips → +3,403). This is CE1's documented
open-then-recover shape reproducing on this object, and the recovery being *synchronous with the
anneal* is direct evidence for §5.1 cause 1: **the rate, not the objective, did the damage.**

**Pre-registered extrapolation, so the last row cannot be narrated after the fact:** if the remaining
600 steps give back a similar fraction as lr anneals to the 2e-07 floor, step 1800 lands near
**+6 to +8%** — still ABOVE the shipped renderer, still a failed epoch, and far from the −10% the
charter's falsifier wanted. CE1's ladder agrees: its aligned runs only crossed BELOW their init after
3,000–6,000 steps. A 1,800-step window at this rate cannot get there.

*(contest-CUDA T4 reference for the same object: 0.00020139)*

**The run is IN FLIGHT at hand-off and this table has one row. Stated plainly rather than padded.**
Measured CPU rate `r = 3.6315 s/step` puts the step-600 row at ~48 min of wall clock plus an 11.4 min
n600 eval, and the full 1,800-step window at ~2.6-3.3 h depending on contention from the co-resident
governed Metal burn. The run is detached, `--nice 10`, writes a preserved EMA checkpoint every 600
steps, and is `--resume-from`-able. Its rows land in `<RUN>/result.json` and its log line for each
eval is `{"step": N, "quantized_exact_seg": ...}`. **Harvest is FO-1 in `$STORE/FIRE_ORDER.sh`**,
which is written, syntax-checked and flag-verified against each tool's real argparse.

I chose to hand off the corrections in Sec 1-2 and Sec 7 rather than hold them for 3 more hours: any
other arm currently training against `gt_n600.npz`, or gating on `d_pose <= 1.25e-4`, is wrong right
now, and those are cheap to act on immediately.

### 5.1 The charter's falsifier FIRED at epoch 1 — and in the worse direction

The charter's falsifier reads "advisory d_seg does not fall >= 10% at any kept epoch". What happened is
stronger than that: d_seg **ROSE 31.23%**, +7,510 flips, **12.4x** CE1's measured 605-flip A/A noise
floor at step 600. That is a real move, not noise, and it costs **+0.006366 S** on the seg leg alone.

I pre-registered in §8 that I expected the falsifier to fire. It did, at the first evaluated epoch.

**What this does and does not close.** It is an **INSTANCE** result — this init, lr 2e-5, an 1,800-step
cosine, seg-only loss — not a family verdict, and I will not narrate it as one. Three causes are live,
and all three were declared as risks in §4 *before* the run:

1. **The LR did not transfer, and the direction of the error is now measured.** CE1's 2e-5 "plateau"
   was measured on a *different* init (d_seg 2.86e-4 against the TOKENS). The shipped renderer is a
   converged object at 2.04e-4, and its own PR130 stage-08 tail LR is **2e-7** (the BS16 receipt:
   −3.03% in 30 steps at batch 16). `hr1` says 2e-7 is "an ancestor anchor, not an automatic value" —
   true, but 2e-5 is equally borrowed, and this run measures which direction the borrowing failed in.
   CE1's own EF0 at 600 steps was ALSO a rise (+636 flips); mine is 11.8x larger, on an object 1.4x
   closer to its optimum. This is the leading suspect.
2. **The target is harder than CE1's.** Training against DALI GT asks the renderer to OVERRIDE its
   input at 9,179 sites; CE1's target was the tokens themselves, which the renderer already reproduces.
3. **τ anneals 21.7x faster per step** in an 1,800-step window than in w96b's 39,000, so this run
   reaches the sharp small-τ regime far earlier in optimization than the schedule was calibrated for.

Causes 1 and 3 are both *"the window and the rate were borrowed"*, which is the same class as §1's
stale identifiers and §2's mislabelled lineage: **a constant that was correct for its own vehicle,
transferred without re-derivation.** That is three independent instances of one failure mode inside a
single charter.

**Not stopped.** The run continues to steps 1,200 and 1,800. CE1's ladder shows the aligned objective
opening with an excursion and recovering only by 3,000–6,000 steps, so the remaining rows measure
whether this excursion turns — at 1,800 steps, probably not, but the rows are nearly free now and
their absence would be a guess.

### 5.2 THE COUPLING — the number this arm existed to produce, MEASURED

`retained/verdict_ft1_step600.json`. n200 **seeded random** draw (seed 20260903, not a prefix),
`[macOS-CPU advisory]`, DALI lineage, through the shipped receiver's composition
(carrier frame 2p + rendered frame 2p+1) and upstream's own SegNet/PoseNet.

| object | d_seg | d_pose | S @ 180,002 B |
|---|---|---|---|
| base (shipped renderer) | 2.00297e-04 | 9.0025e-06 | 0.1493738 |
| **candidate, REALIZED** (export → receiver parse-back) | 2.69623e-04 | **1.507375e-02** | **0.5350674** |
| trained weights (diagnostic, never ships) | 4.44590e-04 | 8.368573e-02 | 1.0791140 |

The base row is the calibration check: d_seg 2.00297e-04 is **0.55%** from the contest-CUDA receipt's
2.0139e-04, and d_pose 9.00e-06 is 1.41x its 6.37e-06 — a CPU-vs-CUDA gap at n200, in the expected
regime. The base is trustworthy, so the candidate rows are too.

> **`coupling_dpose_over_dseg` = |Δd_pose| / |Δd_seg| = 217.30**

**Two independent measurements now agree on the renderer→(seg, pose) coupling of this object:** `rf1`
measured **166.8** for an *un-retrained structural* change; ft1 measures **217.30** for a *trained*
seg-only fine-tune — 1.30x worse, same regime. This is no longer one arm's number.

**The closing arithmetic.** A 25% seg cut is Δd_seg = −5.0348e-05. At coupling 217.30 it drags
Δd_pose = **+1.0941e-02 = 1,718x** the 6.37e-06 base. Apply the measured n600 carrier-recovery
ceiling — and remember a renderer change moves all 600 pairs at once, so **there is no per-pair
admission lever**:

| recovery | post-solve d_pose | × base | × the 1.694e-05 ceiling |
|---|---|---:|---:|
| 5.87× (fcd2) | 1.864e-03 | 293× | **110×** |
| 8.0× (jg5, the best measured) | 1.368e-03 | 215× | **81×** |

**Even a perfect terminal re-solve at the best recovery ever measured lands 81× over the pose budget
that a 25% seg win can pay for.** The measured step-600 candidate is worse still: d_pose 1.507e-02 =
2,366× base, 111× the ceiling after an 8× re-solve, and ΔS = **+0.3857**.

**Verdict.** The **seg-only** aligned renderer fine-tune at this size is **CLOSED by measured
arithmetic**, at `verdict_scope: FORMULATION` — seg-only expected-flip margin loss, no pose term in
the loop, this object, this size. It is closed not because d_seg failed to fall (though it did fail),
but because the coupling makes the pose leg unpayable at ANY useful seg gain.

**What stays OPEN, and why the distinction is real.** 217.30 is the coupling *of the seg-only gradient
direction*. A joint loss does not accept that direction — it explicitly prices pose and searches for a
lower-coupling one. That is exactly the w96b/qbr1 construction, and it is untested on THIS object from
THESE weights. The honest prior is not encouraging: w96b ran pose in-loop from step zero and still
landed d_pose 1.30e-03 = 204× gb1's 6.37e-06, ~80% of its composed delta. But 204× (pose in loop) vs
2,366× (pose absent) is an order of magnitude, and that gap is the whole remaining question.

### 5.3 The realization gap is LARGE — and here it REPAIRED the damage

`trained_vs_realized_max_abs_delta = 0.00232` (nonzero, as §6 predicted). The trained weights score
S 1.0791; the object the receiver actually loads scores S 0.5351. **The deployed encoder threw away
half the damage** — because it discards the 190 of 192 FiLM rows per tensor that training moved, and
training was moving them the wrong way.

That is a striking and uncomfortable fact: **training and export are fighting each other.** Had I
scored the checkpoint's own weights, as the first version of this instrument did, every number above
would have been 2× worse and attributed to the wrong cause. It is also a warning for any successor —
a fine-tune whose gains live in the pruned FiLM rows will have those gains DELETED at export, silently,
at unchanged size.

## 6. THE REALIZATION GAP — and the fix I had to make to my own instrument

My first verdict instrument scored the **checkpoint's** state dict. That is not what ships. The
trainer's QAT models a **uniform int4** grid with **no row prune**; the deployed SM3R encoder quantizes
to the per-tensor depth table {3,4} **and keeps only 2 of 192 rows** in each of
`blocks.{1,2,3}.film.weight`. A trained state that moved all 192 rows has **190 of them discarded at
export**. A d_seg measured on the trained weights is a d_seg for a model that never ships — NO-FAKE
class 8 in miniature, in my own code.

Fixed (commit `8865d5e65`): `export_section` now returns the **shipped receiver's own parse-back
state**, refuses if encoder and receiver disagree by any amount, and the verdict scores that realized
state as the candidate of record. The trained state is kept as a labelled diagnostic and the gap is
reported in d_seg, d_pose and S. Regression-tested: a perturbed state must come back with exactly
`kept_rows` nonzero rows and a nonzero trained-vs-realized delta.

**This is the arm's largest live risk** and it is now instrumented rather than hidden.

Two more defects of my own, found by re-reading my own fixes and fixed with regressions:

* **EMA custody (`0546f657d`).** The loader was a key-preference fallthrough that *happened* to land
  on the shadow. The PR130 trainer declares `deployment_weights="ema_shadow"` and puts the shadow at
  the top-level `state_dict`, with the live weights at `training_state.model_state_dict`
  (verified at source, `:1404` `deployment_state=ema.state_dict()`). The loader now reads the
  declaration and refuses anything else — a silent pick of live weights would have violated the EMA
  non-negotiable invisibly, inside a receipt.
* **Prefix bias (`a279c9c6d`).** `--pairs N` took `range(N)`. On this video a prefix measures the
  POSE axis **2.5–4.2× harder** and seg **~0.96×** — biased in *opposite* directions (memory `m96`).
  The single quantity this arm exists to produce is the RATIO `Δd_pose/Δd_seg`, so a prefix would
  compound both biases exactly where it hurts. Subsets are now a seeded random draw, labelled in the
  receipt, and a prefix is refused by regression.

## 7. THE ARITHMETIC THAT GOVERNS THE FIRE ORDER (re-derived, replaces the charter's)

From the AFR1 receipt, `experiments/results/modal_auth_eval_mirror/contest_auth_eval_modal-ddm_afr1_tile48_groupbin8_cuda_n600_20260831.json`
[contest-CUDA T4 n600]: `d_seg 0.00020139`, `d_pose 6.37e-06`, `180,002 B`, `S 0.14797617125559104`
(seg 0.020139 + pose 0.00798123 + rate 0.11985594).

This arm buys **ΔB = 0**, so `rf1`'s BAR-SEG (`λ_B·|ΔB|/100`) collapses to **0**: the seg leg must
strictly improve and the whole move is paid on distortion.

| seg cut | ΔS_seg | pose ceiling `d_pose_max` | × base |
|---|---:|---:|---:|
| 25% (charter's prediction) | −0.0050348 | **1.694e-05** | 2.66× |
| 10% (charter's falsifier floor) | −0.0020139 | **9.990e-06** | 1.57× |
| charter's guard `d_pose ≤ 1.25e-4` | — | costs **+0.027374 S** | **5.44× the 25% credit** |

**The charter's `d_pose ≤ 1.25e-4` AND-guard cannot be a promote criterion.** It is `af1`'s
whole-gap-to-0.12 budget, imported at the wrong operating point. The binding ceiling is ~1.7e-5.

Working backward through the measured n600 carrier-recovery ceiling (**5.87× fcd2 / 8.0× jg5**, and
this arm has **no per-pair admission lever** — a renderer weight moves all 600 pairs at once), the
**pre-re-solve** `d_pose` must stay **≲ 1.0e-4** for the terminal solve to have any chance.

**And the prior says this is hard:** `rf1` measured `Δd_pose/Δd_seg = 166.8` for an un-retrained
renderer structural change — pose was 97.59% of the damage. If that coupling transfers, a 25% seg cut
costs `Δd_pose ≈ 8.4e-3` = 1,318× base, needing ~496× carrier recovery against a measured mean ceiling
of 5.9–8.0×. **Measuring whether it transfers to a small trained fine-tune is the single most valuable
number this arm can produce**, and nobody has it.

## 8. WHERE THIS ARM SITS RELATIVE TO msr1's CLOSURE (stated, not evaded)

`msr1` closed `FAMILY: boundary-moving-actuators × INSTANCE: dx2 × n600` and its `NEXT_IF_RESUMED`
says *"Do not charter a third boundary-repair actuator on dx2… the manufactured-seg axis on this object
is closed for zero-byte repair."* It names **learned renderer weights** inside that family.

This arm claims to sit at msr1's own registered falsifier, on two measured grounds:

1. **msr1's ledger prices a sign-SHARED move.** It sweeps a uniform logit offset δ over the whole
   1-px shell and measures selectivity 36–52× against the 179.4:1 population ratio it needs. A
   per-pixel expected-flip gradient is not that: each site is pushed toward **its own** target class,
   so the two shell populations receive **opposite-signed** pressure. msr1 states its own falsifier as
   *"any actuator measured to move manufactured deficits without moving correct-pixel margins
   comparably on the same shell."*
2. **The aligned target is a different objective than msr1 modelled.** msr1 priced moving the painted
   boundary relative to the **token** boundary. At the 9,179 positions where token ≠ DALI GT, the
   aligned objective asks the renderer to **override its input**, which is not a boundary move in
   msr1's sense at all.

Neither ground is proof. The honest test is the per-epoch B/H/W selectivity against msr1's measured
36–52× and required 179.4×, which is exactly what the verdict instrument reports.

**Verified at source, not via relay** (`ddm_msr1_manufactured_seg_reduction_20260823.md:17-22`):

> "Any actuator that moves the painted boundary of interface (a,b) in one direction over a region
> fixes the `a→b` flow there and, one for one, deepens the `b→a` flow. Balanced flow inside an
> actuator's addressing cell is unreachable *by construction*, whatever the actuator is: **learned
> renderer weights**, a hand-written palette bias, a class-confidence shift, or a shipped mask."

and its own registered assumption-falsifier (`:366-368`):

> "…the assumption is stated, and it is **falsified by any actuator measured to move manufactured
> deficits without moving correct-pixel margins comparably on the same shell.**"

**Pre-registered, before the result — with the denominator stated, because the floor you divide by
decides the answer.** msr1's flow-balance ceiling is **2,123 net px**. msr1 states that as **6.38% of
the 42,382 B rate demand**; against **d_seg's own 23,757 flips** it is **8.94%**. The second
denominator is the one the charter's falsifier uses, and 8.94% is **below** the charter's 10%
threshold and far below its 25% prediction. The best measured aligned descents anywhere (CE1
EF3000/EF6000) were **6.77%** and **8.16%**. **I expect the charter's falsifier to fire.** Recorded
here, before the rows land, so the outcome cannot be narrated either way.

## 9. TYPED MAIN FIRE ORDER (not run by this arm)

Fire only if the §5 table shows a kept epoch whose **realized** (post-export-parse-back) advisory
d_seg falls and whose realized d_pose stays ≲ 1.0e-4.

```
FO-1  n600 advisory verdict of the candidate through the real receiver
      .venv/bin/python tools/safe_run.py --rss-mb 12000 --timeout 5400 --label ft1_verdict -- \
        .venv/bin/python experiments/ddm_ft1_verdict_bhw_pose.py \
          --candidate <RUN>/ckpt.stage-expected_flip.step00XXXX.full_state.pt \
          --pairs 600 --batch-size 4 --threads 4 --label ft1_epNN \
          --out <RETAINED>/verdict_ft1_epNN.json
      cost ~33 min (3 passes x 660 s). Emits realized d_seg/d_pose, the trained-vs-realized gap,
      B/H/W overall, per GT class, AND split by whether the input token already matched GT
      (the aligned objective's own premise: benefit should land on the 9,179 override sites,
      not as harm on the 117.9M agreeing ones), plus the 36,130 B section + sha.
      Subsets are a seeded random draw; a prefix is refused.
      GATE: size_preserved == true AND parse_back_max_abs_delta == 0.0.

FO-2  terminal pose re-solve on the candidate's renders  (ONLY if FO-1 shows realized d_seg down)
      experiments/ddm_up2_shipping_pose_solve.py solve --gt-cache <DALI> --axis contest_cuda
      (12 int12 coefficients x 600 pairs; 0 archive bytes; the shipped splice costs +45 B measured)
      GATE (DERIVED, replaces the charter's 1.25e-4):
        post-solve d_pose <= 1.694e-05  when realized Delta d_seg <= -5.03e-05 (25% cut)
        post-solve d_pose <=  9.99e-06  when realized Delta d_seg <= -2.01e-05 (10% cut)
        general form: sqrt(10*d_pose_new) < 0.00798123 + 100*|Delta d_seg|
      Expect <= 5.87-8.0x mean recovery (fcd2/jg5). There is NO per-pair admission lever here.

FO-3  splice + T4 buy  (ONLY if FO-2 clears its gate)
      tools/fire_modal_auth_eval.py --seal <candidate archive>
      PROMOTE IFF exact S < 0.14797617125559104 on [contest-CUDA T4 n600].
      Do NOT use the charter's "AND d_pose <= 1.25e-4" conjunct: it admits candidates
      that are 5.44x worse in S than the seg credit they buy. S < AFR1 is the whole gate.
```

## 9b. COMMITS

| sha | what |
|---|---|
| `77af37116` | identity gate + aligned caches + B/H/W verdict + 27 tests |
| `8865d5e65` | score the bytes that ship (realization gap) |
| `0546f657d` | honour the trainer's EMA deployment declaration |
| `ac3cb6215` | this memo |
| `6dedd44b5` | fork correction, 18,954 (inferred) → 20,671 (measured) |
| `a279c9c6d` | seeded random pair draw, never a prefix |

## 10. RECALL EVIDENCE (what I consumed, and what it changed)

| source | what it changed here |
|---|---|
| `ddm_ce1_…` + `semantic_renderer_oracle.py:181-194` | the exact loss law and τ 0.15→0.05; `ce/softplus fraction 0.0`; the A/A noise floor of 605 flips |
| `ddm_w96a/w96b_…` | LR 2e-5 + cosine 1% floor, batch 1, 600 steps/epoch; and the warning that its family CLOSED on **pose**, ~80% of the composed delta, with pose already in-loop |
| `ddm_mst1/msr1/ms9/mf1_…` | 90.4702% manufactured; the 1-px shell (99.66%); hairline deficits (median 0.1621 vs 5.763 correct margin); the 179.4:1 collateral ratio and the 36–52× measured selectivity; §8's family question |
| `ddm_cfa1_…` | only uint8 and argmax are non-analytic — the STE + sigmoid pair is the complete surrogate set, so the charter's "no proxy loss" is *derivable*, not stylistic |
| `ddm_rf1_…` | BAR-SEG = `λ_B·ΔB/100` → **0** at ΔB=0; and the 166.8 renderer coupling in §7 |
| `ddm_tv1/tv2_…` | seg slack and pose damage are co-located; interiors are pose-guarded at 1,039× base → do not shape the loss toward interiors |
| `ddm_jg1/jg5/up2/fcd2_…` | the carrier writes frame 2p and was solved against the ORIGINAL frame 2p+1; recovery ceiling 5.87–8.0× at n600; no admission lever here |
| `ddm_bz2d_…`, `#1142` | the fork factors that made §2's discrepancy legible |
| `ddm_pk4_…` | torch-CPU is the pose authority; no fitted frame-0 overlay |
| `ddm_qs3_…`, `fcd1` | the B/H/W law verbatim, and `157 = 2H + R` — d_seg cannot recover B and H |
| memory `m106`, `binding-instruction-numbers-expire…` | why §1's stale identifiers were predictable |

## 11. DEAD-ENDS AND HONEST LIMITS

- **The trainer has NO pose term.** The charter requires pose in the loop from step zero and forbids a
  pose-free loss. Building it correctly needs the carrier composition (frame 2p from
  basis@coefficients, frame 2p+1 from the renderer) inside the training graph plus a PoseNet
  forward/backward at camera resolution — new, load-bearing, unreviewed code, and roughly 2–3× the
  step cost. **I did not build it.** This run is therefore a **coupling measurement**, explicitly
  MECHANISM-REDUCED, and **cannot produce a family verdict**. It can produce the one number that
  decides whether building the pose term is worth it: the realized Δd_pose/Δd_seg for a small trained
  fine-tune of this exact object.
- The trainer's internal `deployed_argmax_parity` gate uses the **legacy uniform-int4** packer
  (40,252 B), not SM3R mode 6. It does not cover the deployed representation; §1 and §6 do.
- The cost model is an exactly-determined 3-point fit with no out-of-sample check.
- Every d_seg/d_pose here is `[macOS-CPU advisory]`. The step-0 1.23% agreement with the T4 receipt is
  encouraging, not authority.

## 12. LIVE HYPOTHESES

- The per-pixel sign-directed gradient beats msr1's sign-shared selectivity bound (§8). Test: per-epoch
  B/H selectivity vs 36–52× measured / 179.4× required.
- The 9,179 token≠GT positions are reachable by the renderer overriding its input, and are a
  *different* mechanism from boundary jitter. Test: the per-class B/H/W split at those positions.
- The realization gap (§6) is small because the pruned FiLM rows carry little gradient. Test: the
  reported `trained_vs_realized` deltas. If it is large, `--fixed-zero-mask` (or a prune-row-only mask
  added to the trainer) becomes mandatory.

## NEXT_IF_RESUMED

0. The run may still be live: check `pgrep -f train_semantic_quantized_resumable` and
   `<RUN>/run.log`. If it died, `--resume-from <RUN>/ckpt.resume.latest.pt` continues it bit-faithfully.
1. Read `<RUN>/result.json` and the §5 rows. If no kept epoch lowered realized d_seg by ≥10%,
   **count the charter's falsifier as FIRED** and say so plainly; that closes "renderer fine-tune at
   this size, seg-only loss" at FORMULATION scope — not the family, because the pose term was never
   built.
2. Run FO-1 on the best kept checkpoint. The single number to harvest is
   `delta.coupling_dpose_over_dseg`.
3. **DONE — the coupling is 217.30, so the ≳100 branch of this rule fired.** The seg-only formulation
   is closed by arithmetic (§5.2). The next decision is NOT another seg-only LR: it is whether to build
   the pose term (carrier composition + PoseNet at `∂S/∂d_pose = 626.5` on the DALI operating point,
   per af1) and re-run, knowing w96b's pose-in-loop precedent landed 204× over. Price that against
   `ddm_fb1`'s other folds before spending the window.
4. Do not re-run anything against `gt_n600.npz` as a target.

## Own-vehicle frontier

**afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600] — UNMOVED by this arm.**
