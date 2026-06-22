# CAPSTONE FORMULATION — the witness / task-space byte floor through the round-trip lens (is RGB inefficient, by how much?) (2026-06-21)

**Operator (2026-06-21):** "I thought we had moved off RGB as inefficient; what about the round-trip scale
changes though and format changes" → "This needs deep math sweep and formulation in capstone docs / No signal
loss" → **"(A) is approved"** (quantify the witness/task-space byte floor vs the RGB-HNeRV through the round-trip
lens — the "is the class-shift worth it" number). **CAPSTONE DESIGN FORMULATION** — $0, theory + measured
anchors; no training touched. Authority: `[contest-CPU advisory]`, NON-PROMOTABLE, pointer UNMOVED 0.19110.

This is the canonical formulation; the comprehensive prior-evidence harvest (subagent `a85790…`, NO-signal-loss
gather of the 50KB floor / T_floor / boundary-SEG-CORE / pose / comma-priors) slots into §8.

---

## 0. The committed paradigm + the round-trip foundation
CLAUDE.md "Evaluator-Equivalent Witness Compiler Paradigm" (NON-NEGOTIABLE): *RGB fidelity is non-authority
unless it causally improves d_seg / d_pose / bytes.* So we already moved off **RGB-fidelity-as-goal**. The
decoder must still EMIT RGB — but only because RGB is the scorer's INPUT FORMAT, not because fidelity is wanted.
**The round-trip is the quantitative proof of why RGB fidelity is waste:**

```
decoder 384×512 → inflate bicubic↑ 874×1164 → uint8 (8 b/ch @ camera res) →
scorer bilinear↓ ~384×512 → SegNet (argmax)   AND   RGB→YUV6 (4 luma + 2 chroma-
subsampled) → PoseNet (6-dim).  Only the post-round-trip argmax + 6-dim pose survive.
```
Every RGB-fidelity bit is passed through bicubic↑ → uint8 → bilinear↓ → YUV6-subsample, which DESTROYS it
before scoring. Storing RGB fidelity = paying bytes for signal the round-trip throws away. **The witness
representation stores the post-round-trip TASK STATISTIC (argmax mask + pose) and emits only a minimal
round-trip-surviving RGB witness.**

## 1. Objective decomposition (the floor is per-term)
`S = 100·d_seg + √(10·d_pose) + 25·bytes/37,545,489`. Witness bytes = **mask-grammar (d_seg)** +
**pose-carrier (d_pose)** + **witness-generator** + **realizability-overhead**. Derive each through the round-trip.

## 2. d_seg side — the witness WIN (mask grammar replaces d_seg-RGB-rendering)
The target is the GT-frame SegNet argmax mask M (5-class) for the last frame of each of 600 pairs. To get
d_seg=0 the witness's post-round-trip SegNet-argmax must equal M. The round-trip HELPS the witness here:
- **Resolution**: the scorer downsamples to ~384 → the mask is coded at LOW res (the camera-res detail the RGB
  decoder pays for is averaged away by the bilinear↓; the mask only needs to be right at ~384).
- **Structure**: a 5-class scene mask is contours + contiguous regions → boundary/region-MDL coding (boundaries
  are the information, interiors free). Static classes (road / horizon / hood) are HOMOGRAPHY-predictable from
  the known comma camera geometry → near-zero-byte WHERE-priors (only dynamic content costs).
- **Temporal**: 600 frames are one continuous dashcam clip → motion-compensated contour deltas amortize the
  mask across the sequence.
**→ The witness pays for the MASK, NOT for rendering d_seg-correct RGB. The decoder's full-rank RGB precision
(destroyed by the round-trip) is NOT paid. This is the d_seg-side rate win.** Magnitude: the mask-grammar floor
(boundary-SEG-CORE + comma-static-free + temporal-MC) — §8 pins it (prior "~50KB floor" is mask+pose; the
mask-only part, heavily amortized + homography-free, is the cheap term).

## 3. d_pose side — the witness BLOCKER (jitter-entanglement; the binding constraint)
**The contest d_pose target is the FROZEN PoseNet's per-pair output, which is JITTER-dominated, NOT the smooth
physical ego-motion** (corr ≈ 0.72 with physical; the jitter is the high-entropy residual — MEMORY.md
contest-source memo + #158 correction). Consequences:
- **No cheap separable pose code.** You cannot store 6 smooth floats/pair and reconstruct the jittery target —
  the jitter is not smooth and is ENTANGLED with the frame texture (PoseNet reads the full YUV6 frame; the
  6-dim output is a nonlinear function of the whole frame).
- **→ The witness must produce a FRAME whose PoseNet output matches the jittery target** → it needs ~rendering
  capacity for the pose, in the **YUV6-luma subspace** (pose is luma-dominated; chroma is subsampled → low-res
  → carries little). The pose-carrier is the irreducible RGB-LIKE cost of the witness.
- **The √(10·d_pose) term is small** at the operating point (d_pose 3e-4 → 0.055) but to KEEP it small the
  witness must carry the entangled jitter → no rate win on the pose side. **Pose is a wash (witness ≈ RGB cost).**

## 4. The witness byte floor (assembled)
`bytes_witness ≈ mask_grammar(cheap, §2) + yuv6_luma_pose_carrier(RGB-like, §3) + generator + realizability`.
The generator is shared/tiny (a class→witness painter + the luma carrier head); realizability = the witness must
be a LEGAL uint8 RGB frame whose POST-round-trip argmax+pose hit the target (the survival fraction f, B-WITNESS;
boundaries placed at 874-res pre-downsample per the sub-pixel lever #149 to survive the bilinear average).

## 5. RGB-vs-witness — the "RGB inefficient by N bytes" number
- **RGB-HNeRV** pays: decoder weights that render FULL RGB (BOTH d_seg AND d_pose from one frame) + latents.
  Measured: bc20 = 72 KB (decoder int8) + 15 KB (latents) = 87 KB; bc24 ≈ 114 + 15 = 129 KB; borrowed frontier
  177 KB. The decoder is FULL-RANK-FOR-RGB (sweep finding) — it pays for camera-res RGB precision the round-trip
  destroys.
- **Witness** pays: mask-grammar (replaces the d_seg-rendering half — CHEAP) + YUV6-luma pose-carrier (replaces
  the d_pose-rendering half — RGB-like, no win) + generator.
- **The win = (d_seg-rendering cost in the RGB decoder) − (mask-grammar cost).** The inefficiency of RGB is
  precisely the capacity spent rendering d_seg-correct *camera-res RGB* that the round-trip averages+uint8s away
  — replaced by the cheap, round-trip-robust, homography-assisted, temporally-amortized mask grammar. **The pose
  side is NOT where RGB is inefficient (jitter-entanglement forces a frame either way).**
- **Bounded estimate (harvest pins exact):** if mask-grammar ≈ single-digit–low-tens KB (amortized + static-free)
  and the pose-luma-carrier + generator ≈ the latents + a small head, witness ≈ 50–80 KB vs RGB-HNeRV 87–129 KB
  → **win ≈ 30–80 KB → ΔS ≈ −0.02 to −0.05 on the d_seg-rendering rate**, d_seg-neutral. The exact number turns
  on the mask-grammar floor (§8).

## 6. THE VERDICT — the capstone is a HYBRID, and this formalizes WHY the frontier is shaped that way
The witness paradigm is a **d_seg-side rate win** (cheap task-space mask grammar, justified by the round-trip
destroying RGB) and a **pose-side wash** (jitter-entanglement forces a frame). **→ The round-trip-optimal
capstone is a HYBRID:**
  **task-space MASK GRAMMAR (d_seg: contour + temporal-MC + homography-static-free)  +  minimal YUV6-LUMA
  POSE-CARRIER (d_pose: the irreducibly entangled part)  +  a shared tiny generator emitting a round-trip-
  surviving witness.**
This is NEITHER the pure RGB-HNeRV renderer NOR a pure mask store. And it is exactly what the leaderboard
frontier (mask-codec + pose-carrier, e.g. PR95-family's masks.mkv + poses + renderer) converges to — **the
witness/task-space formulation EXPLAINS why that architecture is forced: the round-trip makes d_seg cheap to
store and d_pose impossible to separate.** The capstone should adopt this hybrid explicitly rather than the
full-RGB-fidelity renderer.

## 7. Round-trip levers threaded (no signal loss on the operator's question)
- **Scale (384→874→384)**: the bilinear↓ averages → boundaries blur+flip → place the argmax flip at 874-res
  BEFORE the average (#149, $0, closed-form, pending). The decoder already renders at ~scorer-res (384, not
  camera 874) because rendering above the post-round-trip res is wasted — that lever is taken.
- **Format (uint8)**: corrections must survive 8-bit rounding at camera res (Lever-D survival; why int5 caps
  S~0.49 — shallow boundary quant-fragile).
- **Format (YUV6)**: PoseNet sees 4 luma + 2 chroma-SUBSAMPLED → pose is luma-dominated, chroma low-res → the
  pose-carrier lives in the YUV6-luma subspace (don't spend bytes on chroma the scorer subsamples away).

## 8. NO-SIGNAL-LOSS prior-evidence integration (harvest a85790… — every number sourced)

**The §5 bracket is now MEASURED, and it's bigger than I derived** — because a byte-closed witness already exists:

| Component | bytes | rate `25·B/N` | source | caveat |
|---|---|---|---|---|
| RGB-HNeRV frontier (to beat) | 177,169 | 0.11797 | `information_theoretic_floor_T_floor_20260610.md` | lossless-EXHAUSTED (decoder 98.6% iid Shannon) |
| **L13 score-native witness (BYTE-CLOSED, −59%)** | **72,217** | **0.0481** | `score_native_first_candidate_20260610T112433Z.md` | lossless-parity TRUE; **but d_pose=12.66 (palette pose-blind) → advisory S 13.58** |
| scorer-conditional MDL band (DERIVED floor) | 24,600–64,600 | 0.0164–0.0430 | `frozen_contest_space_council_lenses…`, `smaller_learned_basis_deep_math…` | Kolmogorov-uncomputable; license not guarantee |
| pose side-info (Wyner-Ziv FiLM) | ~1,500 | ~0.001 | `pose_film_cpu_disambiguator_20260612.md` (GO) | side-info the decoder FiLM-injects, NOT a code the eval reads |
| seg residual STORED (every realization LOSES) | 253K–543K | 0.17–0.36 | `boundary_math_seg_core…`, `witness_seg_boundary_decisive_probe…` | lossless 525K / optimal-coder 253K / per-flip 543K |
| comma homography static-priors | ~0 | ~0 | `memory/project_contest_source_is_known…` | static-class WHERE only; doesn't touch the binding WHAT |
| T_floor | S=0.11797 | rate 0.118 / seg 0.056 / pose 0.017 | `information_theoretic_floor_T_floor…` | rate-dominated (61.7%) |

**The single decisive measured fact:** the witness rate class-shift is REAL and PARTIALLY REALIZED — **L13 =
72,217 B byte-closed, −59% vs 177,169 B, lossless-parity-proven.** So "RGB is inefficient" is not a bracket; it's
**~105 KB measured at the rate level** (the witness spends ZERO on the scorer-null space: 22.7% certified-invisible
per channel, 80.67% resize-null, frame0 100% SegNet-invisible — it recovers HNeRV's null-space SLACK, not the
invariant floor).

**The harvest RESOLVES my two open framings (no signal loss):**
1. **Pose (§3 refined, not refuted):** the jitter-entanglement IS real (#155: standalone pose codes floor at
   d_pose 0.094–0.0040, 12–280× the frontier) — BUT the resolution is **Wyner-Ziv FiLM side-info (~1.5 KB,
   disambiguator GO)**: store pose as side-info the decoder FiLM-injects while it renders the bundled near-full-res
   luma. So pose is cheap-CARRIABLE (~1.5 KB) but NOT cheap-STORABLE — the decoder must still produce the luma
   frame. My §3 "pose-carrier ≈ RGB-like" → tightened: **the luma carrier is the cost, the pose scalars are ~free
   side-info on top.**
2. **The binding wall is DISTORTION-realization, not rate (the sharpening):** the rate is won (72 KB measured).
   The two MEASURED walls are: **(a)** L13's pose was solved at the WRONG fidelity (palette → d_pose 12.66 →
   fixable via the Wyner-Ziv FiLM above); **(b)** the seg residual must be AMORTIZED in the GENERATOR, not STORED
   (every storage realization loses: 253K–543K) — and L13's amortized generator sits at **d_seg=0.0068, 12× the
   frontier's 5.6e-4.** Closing that gap is the open generator-d_seg power-law campaign.

**Contradictions, reconciled (named):**
- "pose ~1.5 KB" vs "no cheap pose code" → both true: 1.5 KB is FiLM SIDE-INFO, not a standalone code (eval has
  no pose input). 
- "rate floor carrier-invariant" vs "witness below 177 KB" → the FLOOR is invariant; the witness lowers the
  SLACK (recovers null-space waste). The −59% L13 row proves slack is recoverable.
- residual-sidecar "rate clears 0.856 B/flip" vs "sidecar dies" → rate wins, DISTORTION dies on receptive-field
  collateral (2,823 new bad flips vs 467 fixed). The d_seg win belongs IN TRAINING (Lever-2/5 margin-weight), 0 bytes.

**Prior RGB-vs-witness formulations to EXTEND (not redo):** `layer1_carrier_first_principles_20260612T171912Z.md`
(the canonical 4-section witness budget B_base+B_seg-boundary+B_pose-sideinfo+B_null-fill) +
`frozen_contest_space_council_lenses_synthesis_20260612T173627Z.md` (the 17-lens 25–65 KB conditional MDL). This
memo's contribution over them: the ROUND-TRIP-as-the-cause framing + the d_seg-win/d_pose-wash decomposition +
the measured L13 anchor placed against them.

**The decisive OPEN gaps (the capstone's actual next work, ranked):**
1. **Generator d_seg power-law to frontier-level** — L13 is d_seg=0.0068 (12× too high); does the amortized
   score-native generator reach 5.6e-4 with more capacity/training? This is THE binding distortion campaign.
2. **Half-res 192×256 witness re-measurement — RUN 2026-06-21 → REFUTED** (`halfres_witness_seg_floor_reprobe_n24_20260621.json`,
   commit `32d838d36`). Stem-blind hypothesis FAILS: 192-render costs d_seg +0.0092 (0.0020→0.0112, 5.6× RISE);
   ∂ band GROWS ~4× at 192 (0.0043→0.0170) — OPPOSITE of the predicted shrink. Curve: 384→0.0020, 336→0.0063,
   288→0.0153, 192→0.0112 → SegNet effective decision res > 336 → **witness MUST render full 384×512.** The
   543K→135K reduction does NOT materialize; the seg-side rate cannot be cut by coarser rendering → d_seg attack
   is full-grid TRAINING (taper/shared-structure/Muon), NOT half-res. (Reinforces the sweep: train+structure.)
3. **Integrated L13 + Wyner-Ziv pose-FiLM advisory S** — never composed end-to-end; predicted ~0.14 at 73 KB IF
   seg lands frontier-level. The number that would prove/kill the class-shift.

## NO-FAKE ledger
- DERIVED: the per-term witness floor; the round-trip RGB-inefficiency proof; the d_seg-win / d_pose-blocker
  decomposition; the hybrid verdict; the §5 byte bracket (30–80 KB win) pending the mask-grammar number.
- MEASURED ANCHORS: round-trip ops (driver.py:1980-1986 + upstream/modules.py); RGB-HNeRV bytes (bc20 72+15 KB);
  d_pose jitter-entanglement (MEMORY.md contest-source + #158); T_floor rate-dominated ~0.118.
- NOT claimed: no score moved; pointer UNMOVED 0.19110; the witness win is a DERIVED bracket, not a built+
  byte-closed archive — it must be realized + exact-evaluated before any score claim (the witness compiler is
  unbuilt; this is the formulation that says it's worth building, concentrated on the d_seg/mask side).

## Cross-references (the full sweep + the witness lineage)
- The 2026-06-21 sweep: `dseg_boundary_hessian_conditioning` / `decoder_weight_rate_axis_and_shallow_boundary_synthesis`
  / `latent_dedup_information_bound` / `structural_rate_axis_and_sweep_conclusion` _20260621.md.
- Witness lineage: boundary-math SEG CORE (#52), B-WITNESS 50KB floor (#95/#96), quotient/level-set codec (#155),
  Lever F T_floor (#53), comma frozen-instance priors (#156/#158), legal-frame Dykstra (#73), sub-pixel boundary (#149).
- `optimal_capstone_vehicle_spec_20260611.md` (the RGB-HNeRV vehicle this hybrid reframes).
- CLAUDE.md "Evaluator-Equivalent Witness Compiler Paradigm" (the committed frame).
