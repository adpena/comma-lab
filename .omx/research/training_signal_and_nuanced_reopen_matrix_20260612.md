# Training-signal + nuanced-reopen candidate matrix (2026-06-12)

**Operator directives (binding):** *"are there any other that should be wired up as training time
signal"* + *"what else deserves another analysis and a more nuanced [implementation]"* + *"sometimes
training time / capacity / config is necessary for true signal."*

**The unifying principle.** The decoder backprops through the REAL frozen scorer, so it can LEARN any
scorer-measurable property at **0 bytes**. Therefore: **any correction we STORE (sidecar) or APPLY
post-hoc, that a scorer-trained decoder could learn, should be folded into the training loss.** The
ONLY corrections that legitimately stay post-hoc are the ones the decoder STRUCTURALLY cannot fix —
artifacts injected AFTER the decoder (the uint8/resize round-trip). Conversely, anything measured
crudely / in isolation / mid-basin / under-powered deserves a nuanced full-stack re-analysis (the
Lever-D treatment) before it is believed.

**Master signal — ACTIVE (verified).** `eval_roundtrip` (bicubic↑→uint8-STE round-trip) IS simulated
in the distortion arm's loss (driver.py:23/598/602). So the decoder already learns spatial-round-trip
robustness. This is WHY PR98/T10's residual is likely a true post-hoc rounding artifact, not pure
under-training (the ep2236 test confirms).

## THE MATRIX

| Lever / finding | Already a training signal? | WIRE-AS-TRAINING-SIGNAL candidate? | Deserves NUANCED re-analysis? | Owner / status |
|---|---|---|---|---|
| **Lever 1** rate surrogate | YES | — | minor (true entropy vs proxy) | Partner A (review) |
| **Lever 2** seg surrogate + anneal | YES | — | the anneal schedule + soft_cosine form | Partner A (R9–R11) |
| **Lever 3** pose-FiLM | YES | — | working (d_pose −24% live) | live arm |
| **Lever 4** score-aware QAT | YES (train) | **YES — variable-grid QAT (train at the DEPLOYED coarse grid so the decoder recovers the coarsening distortion, eval_roundtrip-style)** | yes (path b of ITEM B) | **NEW task** |
| **Lever 5** margin-weight | YES | — (it IS the home for boundary corrections) | feed the survival-robust flip ID into its pixel weights | Lever-D-nuanced partner |
| **Variable-level codec** | NO (post-hoc) | **YES — same as Lever-4 path b** | **YES — math-optimal waterfill** | **Partner B (running)** |
| **D1 latent dedup** | NO (lossless) | **YES (NOVEL) — latent-structure-inducing regularizer**: the latents are incompressible because nothing MAKES them cross-pair-structured; add a temporal-smoothness / AR-predictability prior on the latents IN TRAINING so they become dedup-/AR-codeable → unlocks D1 + Cool-Chic-AR as REAL rate wins | the regularizer design | **NEW task (standout)** |
| **Cool-Chic AR-prior** | NO | **YES — same latent-structure regularizer** (the AR prior saturated because the latents have no learned temporal conditional structure; induce it) | — (properly powered as-is) | folds into the NOVEL task |
| **PR98/T10 color bias** | NO (post-hoc) | **MAYBE — color-offset training penalty** IFF the ep2236 test shows under-training (shrinks at convergence); if a true uint8-rounding artifact, stays post-hoc (like PR101) | **YES — ep2236 convergence re-validation** | under-power partner (running) |
| **S12 null-preimage** | NO (lossless, 0-distortion) | NO (deterministic fill) | NO | done |
| **Lever-D seg flip-residual** | NO (sidecar) | **YES — fold into Lever-5** (the flips are the boundary pixels Lever-5 up-weights) | **YES — survival-robust selective** | Lever-D-nuanced partner (running) |
| **Witness seg-boundary** | NO (sidecar) | **YES — fold into training** (the original verdict) | structural; same as Lever-D | folds into Lever-D-nuanced + Lever-5 |
| **Cool-Chic d_seg/pose wall** | n/a (different basis) | — | **YES — more epochs / full-res / capacity** ("still descending, NOT a basis limit") | Track-B (queued) |

## THE STANDOUT NOVEL CANDIDATE — latent-structure-inducing regularizer
D1 proved the base_ch20 latents are ~1.3% above the entropy floor (incompressible) and Cool-Chic-AR
proved the AR prior saturates — BOTH because the latents have **no exploitable cross-pair / temporal
structure**. The training-signal inversion: instead of trying to CODE structure that isn't there, add
a small **training regularizer that INDUCES it** — a temporal-smoothness penalty (adjacent pair-codes
close) or an AR-predictability penalty (each pair-code predictable from its predecessors) on the
latents during training. This makes the latents compressible by construction, at a small distortion
cost the decoder absorbs — unlocking BOTH D1 (dedup) and Cool-Chic-AR (AR coding) as REAL rate wins
that are currently 0. This is the cleanest "training/config for true signal" example in the matrix:
the rate headroom isn't in the CODER, it's in SHAPING the thing being coded.

## PRIORITY (by EV, contention-aware — 4 partners + the arm already running)
1. **NOVEL latent-structure-inducing regularizer** — unlocks two dead rate levers at once; $0 design +
   a small training ablation. HIGHEST new-idea EV. → task.
2. **Variable-grid QAT (Lever-4 + codec path b)** — the training-side fix for the variable codec's
   distortion recovery; folds into the curriculum's existing QAT stages. → task (after Partner B's
   waterfill verdict, since math-optimal-allocation may make retraining unnecessary).
3. **Color-offset training penalty (PR98/T10)** — GATED on the under-power ep2236 verdict (only wire
   it if the win is under-training, not a round-trip artifact).
4. **Boundary-flip → Lever-5 feed (Lever-D/witness)** — the Lever-D-nuanced partner emits the
   survival-robust flip set; Lever-5 consumes it as pixel weights. → fold into that partner's output.

**Holding further dispatch:** 4 partners (A levers, B waterfill, under-power audit, Lever-D nuanced) +
the live arm are already sharing CPU. The two NEW tasks (latent-structure regularizer, variable-grid
QAT) are queued for the next free slot — both are training-side, so they also want the arm further
converged for a real full-stack measurement. No signal lost: every candidate is in this matrix + the
task list.
